/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <cassert>
#include <deque>
#include <mutex>

#include "CPUDependencyData.hpp"
#include "DataAccessRegistration.hpp"
#include <MemoryAllocator.hpp>
#include <ObjectAllocator.hpp>

#include "executors/threads/TaskFinalization.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"
#include "BottomMapEntry.hpp"
#include "TaskDataAccesses.hpp"

#include <InstrumentDependenciesByAccessLinks.hpp>
#include <InstrumentTaskId.hpp>

#include <iostream>
#include <sstream>

#pragma GCC visibility push(hidden)

namespace DataAccessRegistration {
	typedef TaskDataAccesses::bottom_map_t bottom_map_t;

	//! Process all the originators that have become ready
	static inline void processSatisfiedOriginators(
			/* INOUT */ CPUDependencyData &hpDependencyData,
						ComputePlace *computePlace,
						bool fromBusyThread) 
	{
		// NOTE: This is done without the lock held and may be slow since it can enter the scheduler
		for (Task *satisfiedOriginator : hpDependencyData._satisfiedOriginators) {
			assert(satisfiedOriginator != 0);

			ComputePlace *computePlaceHint = nullptr;
			if (computePlace != nullptr) {
				if (computePlace->getType() == satisfiedOriginator->getDeviceType()) {
					computePlaceHint = computePlace;
				}
			}

			Scheduler::addReadyTask(
					satisfiedOriginator,
					computePlaceHint,
					(fromBusyThread ? BUSY_COMPUTE_PLACE_TASK_HINT : SIBLING_TASK_HINT)
			);
		}

		/*
		 * As we use tasks as integral part of our Data Structures, specially the accesses in READ/REDUCTION tasks,
		 * we are responsible for their disposal. Here we destruct and deallocate all the tasks we've determined as
		 * not needed anymore, only if nothing else in the runtime needs the task anymore.
		 * As there is no "task garbage collection", the runtime will only destruct the tasks for us if we mark them as
		 * not needed on the unregisterTaskDataAccesses call, so this takes care on tasks ended anywhere else.
		 */
		for (Task *deletableOriginator : hpDependencyData._deletableOriginators) {
			assert(deletableOriginator != nullptr);

			ComputePlace *computePlaceHint = nullptr;
			if (computePlace != nullptr) {
				if (computePlace->getType() == deletableOriginator->getDeviceType()) {
					computePlaceHint = computePlace;
				}
			}
			TaskFinalization::disposeOrUnblockTask(deletableOriginator, computePlaceHint);
		}

		hpDependencyData._satisfiedOriginators.clear();
		hpDependencyData._deletableOriginators.clear();
	}

	static inline DataAccessType combineTypes(DataAccessType type1, DataAccessType type2) 
	{
		if (type1 == READWRITE_ACCESS_TYPE || type2 == READWRITE_ACCESS_TYPE)
			return READWRITE_ACCESS_TYPE;
		else if (type1 == WRITE_ACCESS_TYPE || type2 == WRITE_ACCESS_TYPE) {
			if(type1 == READ_ACCESS_TYPE || type2 == READ_ACCESS_TYPE)
				return READWRITE_ACCESS_TYPE;
			return WRITE_ACCESS_TYPE;
		} else if (type1 == READ_ACCESS_TYPE && type2 == READ_ACCESS_TYPE)
			return READ_ACCESS_TYPE;
		else
			return WRITE_ACCESS_TYPE; // We don't know, serialize just in case!
	}

	inline void upgradeAccess(DataAccess * access, DataAccessType newType) 
	{
		DataAccessType oldType = access->getType();
		// Duping reductions is incorrect
		assert(oldType != REDUCTION_ACCESS_TYPE && newType != REDUCTION_ACCESS_TYPE);

		access->setType(combineTypes(oldType, newType));
	}

	void registerTaskDataAccess(
			Task *task, DataAccessType accessType, bool weak, void *address, size_t length,
			reduction_type_and_operator_index_t reductionTypeAndOperatorIndex,
			__attribute__((unused)) reduction_index_t reductionIndex) 
	{
		/*
		 * This is called once per access in the task and it's purpose is to initialize our DataAccess structure with the
		 * arguments of this function. No dependency registration is done here, and this call precedes the "registerTaskDataAccesses"
		 * one. All the access structs are constructed in-place in the task array, to prevent allocations.
		 */

		assert(task != nullptr);
		assert(address != nullptr);
		assert(length > 0);

		TaskDataAccesses &accessStruct = task->getDataAccesses();

		DataAccess *accessArray = accessStruct._accessArray;
		void **addressArray = accessStruct._addressArray;
		DataAccess *alreadyExistingAccess = accessStruct.findAccess(address);

		if (alreadyExistingAccess == nullptr) {
			size_t index = accessStruct.currentIndex;

			assert(accessStruct.currentIndex < task->getNumDependencies());

			addressArray[index] = address;
			DataAccess *access = &accessArray[index];

			accessStruct.currentIndex++;

			new(access) DataAccess(accessType, task, weak);

			if (accessType == REDUCTION_ACCESS_TYPE) {
				access->setReductionOperator(reductionTypeAndOperatorIndex);
				access->setReductionLength(length);
			} else {
				assert(!weak); // We don't allow weak accesses just yet.
			}
		} else {
			assert(!weak);
			upgradeAccess(alreadyExistingAccess, accessType);
		}
	}

	void finalizeDataAccess(Task *task, 
							DataAccess *access, 
							void *address, 
							CPUDependencyData &hpDependencyData,
							ComputePlace *computePlace) 
	{
		bool last = false;
		assert(computePlace != nullptr);

		Task *parentTask = task->getParent();
		assert(parentTask != nullptr);

		TaskDataAccesses &parentAccessStruct = parentTask->getDataAccesses();
		assert(!parentAccessStruct.hasBeenDeleted());

		// Are we a bottom task?
		if (access->getType() != READ_ACCESS_TYPE && access->getType() != REDUCTION_ACCESS_TYPE &&
			access->getSuccessor() == nullptr) {
			std::lock_guard<TaskDataAccesses::spinlock_t> guard(parentAccessStruct._lock);

			// We must re-check after the lock
			if (access->getSuccessor() == nullptr) {
				// Update the bottom map!
				bottom_map_t *addresses = &parentAccessStruct._accessMap;
				bottom_map_t::iterator itMap = addresses->find(address);
				assert(itMap != addresses->end()); // If we exist, there must be an entry with our address.

				addresses->erase(itMap);

				last = true;
			}
		}

		/*
		 * All the reduction / read clean up parts where we clear and reclaim memory are designed to be called out of a lock,
		 * and they will take one if they need to touch the bottom map. It is a bit counter intuitive because we don't need,
		 * for example, to care that two tasks are "cleaning up" at the same time. First, because it won't happen with the
		 * _isTop atomic, except with the reductions, but the algorithm accounts for that as only a completeCombineAnd-
		 * DeallocateReduction can actually delete the "bottom" reduction.
		 */

		if (access->getType() == REDUCTION_ACCESS_TYPE) {
			ReductionInfo *reductionInfo = access->getReductionInfo();

			// Not needed in weak reductions, but we don't support them
			reductionInfo->releaseSlotsInUse(((CPU *) computePlace)->getIndex());

			if (access->decreaseTop()) {
				cleanUpTopAccessSuccessors(address, access, parentAccessStruct, hpDependencyData);

				if (reductionInfo->incrementUnregisteredAccesses())
					completeCombineAndDeallocateReduction(reductionInfo);

				__attribute__((unused)) bool remove = task->getDataAccesses().decreaseDeletableCount();
				assert(!remove);
			} else {
				reductionInfo->incrementUnregisteredAccesses();
			}
		} else if (!last && access->getType() != READ_ACCESS_TYPE) {
			Task *successor = access->getSuccessor();
			satisfyNextAccesses(address, hpDependencyData, parentAccessStruct, successor);
		} else if (access->getType() == READ_ACCESS_TYPE && access->decreaseTop()) {
			// We were the top. We have to cascade until we find a non-finished access.
			cleanUpTopAccessSuccessors(address, access, parentAccessStruct, hpDependencyData);
			task->getDataAccesses().decreaseDeletableCount();
		}	
	}

	void satisfyNextAccesses(void *address, CPUDependencyData &hpDependencyData, 
							TaskDataAccesses &parentAccessStruct, Task *successor) 
	{
		if (successor != nullptr) {
			DataAccess *next = successor->getDataAccesses().findAccess(address);
			assert(next != nullptr);

			if (next->getType() != REDUCTION_ACCESS_TYPE && successor->decreasePredecessors()) {
				hpDependencyData._satisfiedOriginators.push_back(successor);
			}

			if (next->getType() == REDUCTION_ACCESS_TYPE && next->decreaseTop()) {
				decreaseDeletableCountOrDelete(successor, hpDependencyData._deletableOriginators);
				cleanUpTopAccessSuccessors(address, next, parentAccessStruct, hpDependencyData);
			} else if (next->getType() == READ_ACCESS_TYPE) {
				next->decreaseTop();
				satisfyReadSuccessors(address, next, parentAccessStruct, hpDependencyData._satisfiedOriginators);
			} else {
				if (next->getSuccessor() == nullptr) {
					std::lock_guard<TaskDataAccesses::spinlock_t> guard(parentAccessStruct._lock);
					if (next->getSuccessor() == nullptr) {
						parentAccessStruct._accessMap.find(address)->second.satisfied = true;
					}
				}
			}
		}
	}

	void cleanUpTopAccessSuccessors(void *address, DataAccess *pAccess, TaskDataAccesses &accesses,
									CPUDependencyData &hpDependencyData) 
	{
		DataAccessType accessType = pAccess->getType();
		ReductionInfo *reductionInfo = pAccess->getReductionInfo();

		assert(accessType != REDUCTION_ACCESS_TYPE || reductionInfo != nullptr);

		while (true) {
			Task *successor = pAccess->getSuccessor();
			assert(successor != pAccess->getOriginator());

			if (successor != nullptr) {
				DataAccess *next = successor->getDataAccesses().findAccess(address);
				assert(next != nullptr);

				if (next->getType() == accessType &&
					(accessType != REDUCTION_ACCESS_TYPE || next->getReductionInfo() == reductionInfo)) {
					if (next->decreaseTop()) {
						// Deletable
						decreaseDeletableCountOrDelete(successor, hpDependencyData._deletableOriginators);
						pAccess = next;
					} else {
						// Next one is top. Stop here.
						return;
					}
				} else {
					if(accessType == REDUCTION_ACCESS_TYPE && reductionInfo->finished()) {
						completeCombineAndDeallocateReduction(reductionInfo);
					}
					// Unlock next access and return
					satisfyNextAccesses(address, hpDependencyData, accesses, successor);
					return;
				}
			} else {
				// Could be false positive. We need a lock on the bottomMap.
				std::lock_guard<TaskDataAccesses::spinlock_t> guard(accesses._lock);
				if (pAccess->getSuccessor() == nullptr) {
					bottom_map_t::iterator itMap = accesses._accessMap.find(address);
					assert(itMap != accesses._accessMap.end());

					if(accessType == REDUCTION_ACCESS_TYPE) {
						if (reductionInfo->finished()) {
							completeCombineAndDeallocateReduction(reductionInfo);
							accesses._accessMap.erase(itMap);
						} else {
							itMap->second.access = nullptr;
						}
					} else {
						accesses._accessMap.erase(itMap);
					}

					return;
				}
				// Try again
			}
		}
	}

	void satisfyReadSuccessors(void *address, DataAccess *pAccess, TaskDataAccesses &accesses,
							   CPUDependencyData::satisfied_originator_list_t &satisfiedOriginators) 
	{
		while (true) {
			Task *successor = pAccess->getSuccessor();
			if (successor != nullptr) {
				DataAccess *next = successor->getDataAccesses().findAccess(address);
				if (next->getType() == READ_ACCESS_TYPE) {
					if (successor->decreasePredecessors())
						satisfiedOriginators.push_back(successor);
					pAccess = next;
				} else {
					return;
				}
			} else {
				// Could be false positive. We need a lock on the bottomMap.
				std::lock_guard<TaskDataAccesses::spinlock_t> guard(accesses._lock);
				if (pAccess->getSuccessor() == nullptr) {
					bottom_map_t::iterator itMap = accesses._accessMap.find(address);
					assert(itMap != accesses._accessMap.end());
					itMap->second.satisfied = true;
					return;
				}

				// Try again
			}
		}
	}

	bool registerTaskDataAccesses(Task *task, ComputePlace *computePlace, CPUDependencyData &hpDependencyData) 
	{
		/*
		 * This is only called once per task. It is responsible for registering all the dependencies of the accesses
		 * in the task. It is called after everything has been constructed in the "registerTaskDataAccess" function.
		 * In this implementation, we will go through all the accesses in the access array and we will check and
		 * register the dependencies for each one.
		 */

		assert(task != nullptr);
		assert(computePlace != nullptr);

		// Enable the wait clause to release the dependencies once all children finish
		task->setDelayedRelease(true);

		nanos6_task_info_t *taskInfo = task->getTaskInfo();
		assert(taskInfo != 0);

		task->increasePredecessors(2);

		// This part creates the DataAccesses and inserts it to dependency system
		taskInfo->register_depinfo(task->getArgsBlock(), task);

		TaskDataAccesses &accessStructures = task->getDataAccesses();

		insertAccesses(task, hpDependencyData);

		assert(!accessStructures.hasBeenDeleted());
		if (accessStructures.hasDataAccesses()) {
			task->increaseRemovalBlockingCount();
		}

		processSatisfiedOriginators(hpDependencyData, computePlace, true);

		return task->decreasePredecessors(2);
	}

	void unregisterTaskDataAccesses(Task *task, ComputePlace *computePlace,
									CPUDependencyData &hpDependencyData, 
									__attribute__((unused)) MemoryPlace *location, bool fromBusyThread) 
	{
		assert(task != nullptr);

		TaskDataAccesses &accessStruct = task->getDataAccesses();
		assert(!accessStruct.hasBeenDeleted());

		if (!accessStruct.hasDataAccesses()) return;

#ifndef NDEBUG
		{
			bool alreadyTaken = false;
			assert(hpDependencyData._inUse.compare_exchange_strong(alreadyTaken, true));
		}
#endif
		{
			// if REDUCTION_ACCESS_TYPE, release slot.
			// if task is final, it is always strong. if task is taskloop, it is always strong because taskloop is always final.
			// if task is not final, it may be strong anyway, so we have to check the access itself.
			assert(accessStruct.hasDataAccesses());
			for (size_t i = 0; i < accessStruct.getRealAccessNumber(); ++i) {
				void *raw_address = accessStruct._addressArray[i];
				DataAccess *access = &accessStruct._accessArray[i];
				finalizeDataAccess(task, access, raw_address, hpDependencyData, computePlace);
			}

			/*
			 * All TaskDataAccesses have a deletableCount of 1 for default, so this will return true unless
			 * some read/reduction accesses have increased this as well because the task cannot be deleted yet.
			 */
			if (accessStruct.decreaseDeletableCount())
				task->decreaseRemovalBlockingCount();
		}

		processSatisfiedOriginators(hpDependencyData, computePlace, fromBusyThread);

#ifndef NDEBUG
		{
			bool alreadyTaken = true;
			assert(hpDependencyData._inUse.compare_exchange_strong(alreadyTaken, false));
		}
#endif
	}

	void handleEnterBlocking(Task *task) 
	{
		assert(task != nullptr);

		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		if (accessStructures.hasDataAccesses()) {
			task->decreaseRemovalBlockingCount();
		}
	}

	void handleExitBlocking(Task *task) 
	{
		assert(task != nullptr);

		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		if (accessStructures.hasDataAccesses()) {
			task->increaseRemovalBlockingCount();
		}
	}

	void handleEnterTaskwait(Task *task, ComputePlace *computePlace, CPUDependencyData &dependencyData) 
	{
		assert(task != nullptr);

		TaskDataAccesses &accessStruct = task->getDataAccesses();
		assert(!accessStruct.hasBeenDeleted());

		std::lock_guard<TaskDataAccesses::spinlock_t> guard(accessStruct._lock);
		bottom_map_t &bottomMap = accessStruct._accessMap;

		for (bottom_map_t::iterator itMap = bottomMap.begin(); itMap != bottomMap.end(); itMap++) {
			ReductionInfo *reductionInfo = itMap->second.reductionInfo;

			if(reductionInfo != nullptr) {
				assert(!reductionInfo->finished());
				if(itMap->second.access == nullptr && reductionInfo->markAsClosed())
					completeCombineAndDeallocateReduction(reductionInfo);
				else
					reductionInfo->markAsClosed();

				itMap->second.reductionInfo = nullptr;
			}
		}

		processSatisfiedOriginators(dependencyData, computePlace, false);

		if (accessStruct.hasDataAccesses()) {
			task->decreaseRemovalBlockingCount();
		}
	}


	void handleExitTaskwait(Task *task, __attribute__((unused)) ComputePlace *computePlace,
							__attribute__((unused)) CPUDependencyData &dependencyData) 
	{
		assert(task != nullptr);

		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		if (accessStructures.hasDataAccesses()) {
			task->increaseRemovalBlockingCount();
		}
	}

	void handleTaskRemoval(__attribute__((unused)) Task *task, __attribute__((unused)) ComputePlace *computePlace) 
	{
		// Only needed for weak accesses (not implemented)
	}

	void insertAccesses(Task *task, __attribute__((unused)) CPUDependencyData &hpDependencyData) 
	{
		TaskDataAccesses &accessStruct = task->getDataAccesses();
		assert(!accessStruct.hasBeenDeleted());

		Task *parentTask = task->getParent();
		assert(parentTask != nullptr);

		TaskDataAccesses &parentAccessStruct = parentTask->getDataAccesses();
		assert(!parentAccessStruct.hasBeenDeleted());

		// Default deletableCount of 1.
		accessStruct.increaseDeletableCount();

		size_t reduction_index = 0;

		// Get all seqs
		for (size_t i = 0; i < accessStruct.getRealAccessNumber(); ++i) {
			void *raw_address = accessStruct._addressArray[i];
			DataAccess *access = &accessStruct._accessArray[i];
			DataAccessType accessType = access->getType();
			DataAccess *predecessor = nullptr;
			bool becameUnsatisfied = false;
			ReductionInfo *reductionInfo = nullptr;
			bool closeReduction = false;
			bottom_map_t::iterator itMap;

			// Instrumentation mock(for now)
			DataAccessRegion mock;
			Instrument::data_access_id_t dataAccessInstrumentationId = Instrument::createdDataAccess(
					nullptr,
					accessType, false, mock,
					false, false, false, Instrument::access_object_type_t::regular_access_type,
					task->getInstrumentationTaskId()
			);

			access->setInstrumentationId(dataAccessInstrumentationId);

			bottom_map_t *addresses = &parentAccessStruct._accessMap;
			{
				std::lock_guard<TaskDataAccesses::spinlock_t> guard(parentAccessStruct._lock);

				/*
				 *  Determine our predecessor safely, and maybe insert ourselves to the map.
				 */

				itMap = addresses->find(raw_address);
				if (itMap != addresses->end()) {
					predecessor = itMap->second.access; // This can still be null!
					itMap->second.access = access;
				} else {
					// Insert task to map.
					itMap = addresses->emplace(std::piecewise_construct, std::forward_as_tuple(raw_address),
											   std::forward_as_tuple(access, task)).first;
				}

				/*
				 * Check if we're closing a reduction, or allocate one in case we need it.
				 */

				if (accessType == REDUCTION_ACCESS_TYPE) {
					ReductionInfo * currentReductionInfo = itMap->second.reductionInfo;
					reductionInfo = currentReductionInfo;
					reduction_type_and_operator_index_t typeAndOpIndex = access->getReductionOperator();
					size_t length = access->getReductionLength();
					if (currentReductionInfo == nullptr || currentReductionInfo->getTypeAndOperatorIndex() != typeAndOpIndex ||
							currentReductionInfo->getOriginalLength() != length) {
						currentReductionInfo = allocateReductionInfo(accessType, reduction_index++, typeAndOpIndex,
															  raw_address, length, *task);
						currentReductionInfo->incrementRegisteredAccesses();
					} else {
						currentReductionInfo->incrementRegisteredAccesses();
					}

					itMap->second.reductionInfo = currentReductionInfo;

					assert(currentReductionInfo != nullptr);
					assert(currentReductionInfo->getTypeAndOperatorIndex() == typeAndOpIndex);
					assert(currentReductionInfo->getOriginalLength() == length);
					assert(currentReductionInfo->getOriginalAddress() == raw_address);

					access->setReductionInfo(currentReductionInfo);

					accessStruct.increaseDeletableCount();
				} else {
					reductionInfo = itMap->second.reductionInfo;
					itMap->second.reductionInfo = nullptr;

					if(accessType == READ_ACCESS_TYPE)
						accessStruct.increaseDeletableCount();
				}

				/*
				 * Check if we are satisfied
				 */

				if (predecessor != nullptr) {
					DataAccessType predecessorType = predecessor->getType();

					if(predecessorType == READ_ACCESS_TYPE) {
						becameUnsatisfied = !(accessType == READ_ACCESS_TYPE && itMap->second.satisfied);
					} else if(predecessorType == REDUCTION_ACCESS_TYPE) {
						assert(!predecessor->closesReduction());
						becameUnsatisfied = true;

						if(accessType != REDUCTION_ACCESS_TYPE || predecessor->getReductionInfo() != access->getReductionInfo())
							predecessor->setClosesReduction(true);
					} else {
						becameUnsatisfied = true;
					}

					// Reductions always start
					if(accessType == REDUCTION_ACCESS_TYPE)
						becameUnsatisfied = false;

					predecessor->setSuccessor(task);

					itMap->second.satisfied = !becameUnsatisfied;
					if (becameUnsatisfied) task->increasePredecessors();
				} else {
					if (accessType == READ_ACCESS_TYPE || accessType == REDUCTION_ACCESS_TYPE)
						access->decreaseTop();

					if(reductionInfo != nullptr && access->getReductionInfo() != reductionInfo) {
						closeReduction = reductionInfo->markAsClosed();
						assert(closeReduction);
					}
				}
			} // Lock Release

			if(closeReduction) {
				completeCombineAndDeallocateReduction(reductionInfo);
			}

			if (!becameUnsatisfied) {
				Instrument::dataAccessBecomesSatisfied(
						dataAccessInstrumentationId,
						true,
						task->getInstrumentationTaskId()
				);
			}
		}
	}

	void releaseReductionInfo(ReductionInfo *reductionInfo) 
	{
		// Reduction must have finished to be combined and destroyed.
		assert(reductionInfo->finished());

		__attribute__((unused)) bool wasLastCombination = reductionInfo->combine(true);
		ObjectAllocator<ReductionInfo>::deleteObject(reductionInfo);

		assert(wasLastCombination);
	}

	void completeCombineAndDeallocateReduction(ReductionInfo *info) 
	{
		assert(info != nullptr);
		assert(info != info->getOriginalAddress());
		assert(info->finished());

		// This will combine the reduction
		releaseReductionInfo(info);
	}

	void
	decreaseDeletableCountOrDelete(Task *originator,
								   CPUDependencyData::deletable_originator_list_t &deletableOriginators) 
	{
		if (originator->getDataAccesses().decreaseDeletableCount() && originator->decreaseRemovalBlockingCount()) {
			deletableOriginators.push_back(originator); // hint the runtime to delete the task.
		}
	}

	ReductionInfo *allocateReductionInfo(
			__attribute__((unused)) DataAccessType &dataAccessType, reduction_index_t reductionIndex,
			reduction_type_and_operator_index_t reductionTypeAndOpIndex,
			void *address, const size_t length, const Task &task) 
	{
		assert(dataAccessType == REDUCTION_ACCESS_TYPE);

		nanos6_task_info_t *taskInfo = task.getTaskInfo();
		assert(taskInfo != nullptr);

		ReductionInfo *newReductionInfo = ObjectAllocator<ReductionInfo>::newObject(
				address, length,
				reductionTypeAndOpIndex,
				taskInfo->reduction_initializers[reductionIndex],
				taskInfo->reduction_combiners[reductionIndex]);

		return newReductionInfo;
	}

	// Placeholder
	void combineTaskReductions(__attribute__((unused)) Task *task, __attribute__((unused)) ComputePlace *computePlace) 
	{
	}
	
	// Placeholder
	void releaseTaskwaitFragment(
		__attribute__((unused)) Task *task,
		__attribute__((unused)) ComputePlace *computePlace,
		__attribute__((unused)) CPUDependencyData &hpDependencyData) 
	{
	}
}

#pragma GCC visibility pop

