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

#include "BottomMapEntry.hpp"
#include "CPUDependencyData.hpp"
#include "DataAccessRegistration.hpp"
#include "executors/threads/TaskFinalization.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "scheduling/Scheduler.hpp"
#include "TaskDataAccesses.hpp"
#include "tasks/Task.hpp"

#include <InstrumentDependenciesByAccessLinks.hpp>
#include <InstrumentTaskId.hpp>
#include <MemoryAllocator.hpp>
#include <ObjectAllocator.hpp>

#pragma GCC visibility push(hidden)

namespace DataAccessRegistration {
	typedef TaskDataAccesses::bottom_map_t bottom_map_t;
	
	void insertAccesses(Task * task);
	
	ReductionInfo * allocateReductionInfo(
		DataAccessType &dataAccessType, reduction_index_t reductionIndex,
		reduction_type_and_operator_index_t reductionTypeAndOpIndex,
		void * address, const size_t length, const Task &task);
	
	void satisfyReadSuccessors(void *address, DataAccess *pAccess, TaskDataAccesses &accesses,
		CPUDependencyData::satisfied_originator_list_t &satisfiedOriginators);
	
	void cleanUpTopAccessSuccessors(void *address, DataAccess *pAccess, TaskDataAccesses &parentAccesses,
		CPUDependencyData &hpDependencyData);
	
	void releaseReductionInfo(ReductionInfo *info);
	
	void satisfyNextAccesses(void *address, CPUDependencyData &hpDependencyData,
		TaskDataAccesses &parentAccessStruct, Task *successor);
	
	void decreaseDeletableCountOrDelete(Task *originator,
		CPUDependencyData::deletable_originator_list_t &deletableOriginators);
	
	
	//! Process all the originators that have become ready
	static inline void processSatisfiedOriginators(
		CPUDependencyData &hpDependencyData,
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
		
		// As we use tasks as integral part of our Data Structures, specially the accesses in READ/REDUCTION tasks,
		// we are responsible for their disposal. Here we destruct and deallocate all the tasks we've determined as
		// not needed anymore, only if nothing else in the runtime needs the task anymore.
		// As there is no "task garbage collection", the runtime will only destruct the tasks for us if we mark them as
		// not needed on the unregisterTaskDataAccesses call, so this takes care on tasks ended anywhere else.
		
		for (Task *deletableOriginator : hpDependencyData._deletableOriginators) {
			assert(deletableOriginator != nullptr);
			TaskFinalization::disposeOrUnblockTask(deletableOriginator, computePlace);
		}
		
		hpDependencyData._satisfiedOriginators.clear();
		hpDependencyData._deletableOriginators.clear();
	}
	
	static inline DataAccessType combineTypes(DataAccessType type1, DataAccessType type2)
	{
		if (type1 == type2) {
			return type1;
		}
		return READWRITE_ACCESS_TYPE;
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
			reduction_index_t reductionIndex)
	{
		// This is called once per access in the task and it's purpose is to initialize our DataAccess structure with the
		// arguments of this function. No dependency registration is done here, and this call precedes the "registerTaskDataAccesses"
		// one. All the access structs are constructed in-place in the task array, to prevent allocations.
		
		assert(task != nullptr);
		assert(address != nullptr);
		assert(length > 0);
		assert(!weak);
		
		TaskDataAccesses &accessStruct = task->getDataAccesses();
		
		DataAccess *accessArray = accessStruct._accessArray;
		void **addressArray = accessStruct._addressArray;
		DataAccess *alreadyExistingAccess = accessStruct.findAccess(address);
		
		if (alreadyExistingAccess == nullptr) {
			size_t index = accessStruct._currentIndex;
			
			assert(accessStruct._currentIndex < accessStruct._maxDeps);
			
			addressArray[index] = address;
			DataAccess *access = &accessArray[index];
			
			accessStruct._currentIndex++;
			
			new(access) DataAccess(accessType, task, weak);
			
			if (accessType == REDUCTION_ACCESS_TYPE) {
				access->setReductionOperator(reductionTypeAndOperatorIndex);
				access->setReductionLength(length);
				access->setReductionIndex(reductionIndex);
			}
		} else {
			upgradeAccess(alreadyExistingAccess, accessType);
		}
	}
	
	inline bool hasNoDelayedRemoval(DataAccessType type) {
		return (type != READ_ACCESS_TYPE && type != REDUCTION_ACCESS_TYPE);
	}
	
	void finalizeDataAccess(Task *task,
		DataAccess *access,
		void *address,
		CPUDependencyData &hpDependencyData,
		ComputePlace *computePlace)
	{
		bool last = false;
		DataAccessType accessType = access->getType();
		assert(computePlace != nullptr);
		
		Task *parentTask = task->getParent();
		assert(parentTask != nullptr);
		
		TaskDataAccesses &parentAccessStruct = parentTask->getDataAccesses();
		assert(!parentAccessStruct.hasBeenDeleted());
		
		if(hasNoDelayedRemoval(accessType))
			task->getDataAccesses().decreaseDeletableCount();
		
		// Are we a bottom task?
		if (accessType != READ_ACCESS_TYPE && accessType != REDUCTION_ACCESS_TYPE &&
			access->getSuccessor() == nullptr) {
			std::lock_guard<TaskDataAccesses::spinlock_t> guard(parentAccessStruct._lock);
			
			// We must re-check after the lock
			if (access->getSuccessor() == nullptr) {
				// Update the bottom map!
				bottom_map_t &addresses = parentAccessStruct._subaccessBottomMap;
				addresses.erase(address);
				last = true;
			}
		}
		
		// All the reduction / read clean up parts where we clear and reclaim memory are designed to be called out of a lock,
		// and they will take one if they need to touch the bottom map. It is a bit counter intuitive because we don't need,
		// for example, to care that two tasks are "cleaning up" at the same time. First, because it won't happen with the
		// _isTop atomic, except with the reductions, but the algorithm accounts for that as only a completeCombineAnd-
		// DeallocateReduction can actually delete the "bottom" reduction.
		
		if (accessType == REDUCTION_ACCESS_TYPE) {
			ReductionInfo *reductionInfo = access->getReductionInfo();
			if (reductionInfo->incrementUnregisteredAccesses())
				releaseReductionInfo(reductionInfo);
			
			if (access->markAsFinished()) {
				cleanUpTopAccessSuccessors(address, access, parentAccessStruct, hpDependencyData);
				
				__attribute__((unused)) bool remove = task->getDataAccesses().decreaseDeletableCount();
				assert(!remove);
			}
		} else if (!last && accessType != READ_ACCESS_TYPE) {
			Task *successor = access->getSuccessor();
			satisfyNextAccesses(address, hpDependencyData, parentAccessStruct, successor);
		} else if (accessType == READ_ACCESS_TYPE && access->markAsFinished()) {
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
			
			if (next->getType() == REDUCTION_ACCESS_TYPE) {
				ReductionInfo * reductionInfo = next->getReductionInfo();
				
				if(reductionInfo->incrementUnregisteredAccesses())
					releaseReductionInfo(reductionInfo);
				
				if(next->markAsTop()) {
					decreaseDeletableCountOrDelete(successor, hpDependencyData._deletableOriginators);
					cleanUpTopAccessSuccessors(address, next, parentAccessStruct, hpDependencyData);
				}
			} else if (next->getType() == READ_ACCESS_TYPE) {
				next->markAsTop();
				satisfyReadSuccessors(address, next, parentAccessStruct, hpDependencyData._satisfiedOriginators);
			} else {
				if (next->getSuccessor() == nullptr) {
					std::lock_guard<TaskDataAccesses::spinlock_t> guard(parentAccessStruct._lock);
					if (next->getSuccessor() == nullptr) {
						parentAccessStruct._subaccessBottomMap.find(address)->second._satisfied = true;
					}
				}
			}
		}
	}
	
	void cleanUpTopAccessSuccessors(void *address, DataAccess *pAccess, TaskDataAccesses &parentAccesses,
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
					if (next->markAsTop()) {
						// Deletable
						decreaseDeletableCountOrDelete(successor, hpDependencyData._deletableOriginators);
						pAccess = next;
					} else {
						// Next one is top. Stop here.
						return;
					}
				} else {
					// Unlock next access and return
					satisfyNextAccesses(address, hpDependencyData, parentAccesses, successor);
					return;
				}
			} else {
				// Lock the bottom map and re-check (in case of races)
				std::lock_guard<TaskDataAccesses::spinlock_t> guard(parentAccesses._lock);
				if (pAccess->getSuccessor() == nullptr) {
					bottom_map_t::iterator itMap = parentAccesses._subaccessBottomMap.find(address);
					assert(itMap != parentAccesses._subaccessBottomMap.end());
					
					if (accessType == REDUCTION_ACCESS_TYPE) {
						if (reductionInfo->finished()) {
							parentAccesses._subaccessBottomMap.erase(itMap);
						} else {
							itMap->second._access = nullptr;
						}
					} else {
						parentAccesses._subaccessBottomMap.erase(itMap);
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
					bottom_map_t::iterator itMap = accesses._subaccessBottomMap.find(address);
					assert(itMap != accesses._subaccessBottomMap.end());
					itMap->second._satisfied = true;
					return;
				}
				
				// Try again
			}
		}
	}
	
	bool registerTaskDataAccesses(Task *task, ComputePlace *computePlace, __attribute__((unused)) CPUDependencyData &hpDependencyData)
	{
		// This is called once per task, and will create all the dependencies in register_depinfo, to later insert
		// them into the chain in the insertAccesses call.
		
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
		
		insertAccesses(task);
		
		assert(!accessStructures.hasBeenDeleted());
		if (accessStructures.hasDataAccesses()) {
			task->increaseRemovalBlockingCount();
		}
		
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
				void *address = accessStruct._addressArray[i];
				DataAccess *access = &accessStruct._accessArray[i];
				finalizeDataAccess(task, access, address, hpDependencyData, computePlace);
			}
			
			// All TaskDataAccesses have a deletableCount of 1 for default, so this will return true unless
			// some read/reduction accesses have increased this as well because the task cannot be deleted yet.
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
		bottom_map_t &bottomMap = accessStruct._subaccessBottomMap;
		
		for (bottom_map_t::iterator itMap = bottomMap.begin(); itMap != bottomMap.end(); itMap++) {
			ReductionInfo *reductionInfo = itMap->second._reductionInfo;
			
			if (reductionInfo != nullptr) {
				assert(!reductionInfo->finished());
				if (itMap->second._access == nullptr && reductionInfo->markAsClosed())
					releaseReductionInfo(reductionInfo);
				else
					reductionInfo->markAsClosed();
				
				itMap->second._reductionInfo = nullptr;
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
	
	void insertAccesses(Task *task)
	{
		TaskDataAccesses &accessStruct = task->getDataAccesses();
		assert(!accessStruct.hasBeenDeleted());
		
		Task *parentTask = task->getParent();
		assert(parentTask != nullptr);
		
		TaskDataAccesses &parentAccessStruct = parentTask->getDataAccesses();
		assert(!parentAccessStruct.hasBeenDeleted());
		
		// Default deletableCount of 1.
		accessStruct.increaseDeletableCount();
		
		// Get all seqs
		for (size_t i = 0; i < accessStruct.getRealAccessNumber(); ++i) {
			void *address = accessStruct._addressArray[i];
			DataAccess *access = &accessStruct._accessArray[i];
			DataAccessType accessType = access->getType();
			DataAccess *predecessor = nullptr;
			bool isSatisfied = true;
			ReductionInfo *reductionInfo = nullptr;
			bool closeReduction = false;
			bottom_map_t::iterator itMap;
			
			// Instrumentation mock(for now)
			DataAccessRegion mock(address, 1);
			Instrument::data_access_id_t dataAccessInstrumentationId = Instrument::createdDataAccess(
					nullptr,
					accessType, false, mock,
					false, false, false, Instrument::access_object_type_t::regular_access_type,
					task->getInstrumentationTaskId()
			);
			
			accessStruct.increaseDeletableCount();
			access->setInstrumentationId(dataAccessInstrumentationId);
			
			bottom_map_t &addresses = parentAccessStruct._subaccessBottomMap;
			{
				std::lock_guard<TaskDataAccesses::spinlock_t> guard(parentAccessStruct._lock);
				
				//Determine our predecessor safely, and maybe insert ourselves to the map.
				itMap = addresses.find(address);
				if (itMap != addresses.end()) {
					predecessor = itMap->second._access; // This can still be null!
					itMap->second._access = access;
				} else {
					// Insert task to map.
					itMap = addresses.emplace(std::piecewise_construct, std::forward_as_tuple(address),
											   std::forward_as_tuple(access)).first;
				}
				
				// Check if we're closing a reduction, or allocate one in case we need it.
				if (accessType == REDUCTION_ACCESS_TYPE) {
					ReductionInfo * currentReductionInfo = itMap->second._reductionInfo;
					reductionInfo = currentReductionInfo;
					reduction_type_and_operator_index_t typeAndOpIndex = access->getReductionOperator();
					size_t length = access->getReductionLength();
					
					if (currentReductionInfo == nullptr || currentReductionInfo->getTypeAndOperatorIndex() != typeAndOpIndex ||
						currentReductionInfo->getOriginalLength() != length) {
						currentReductionInfo = allocateReductionInfo(accessType, access->getReductionIndex(), typeAndOpIndex,
							address, length, *task);
						if(predecessor == nullptr)
							currentReductionInfo->incrementUnregisteredAccesses();
					}
					
					currentReductionInfo->incrementRegisteredAccesses();
					itMap->second._reductionInfo = currentReductionInfo;
					
					assert(currentReductionInfo != nullptr);
					assert(currentReductionInfo->getTypeAndOperatorIndex() == typeAndOpIndex);
					assert(currentReductionInfo->getOriginalLength() == length);
					assert(currentReductionInfo->getOriginalAddress() == address);
					
					access->setReductionInfo(currentReductionInfo);
				} else {
					reductionInfo = itMap->second._reductionInfo;
					itMap->second._reductionInfo = nullptr;
				}
				
				// Check if we are satisfied
				if (predecessor != nullptr) {
					DataAccessType predecessorType = predecessor->getType();
					
					if (predecessorType == READ_ACCESS_TYPE) {
						isSatisfied = (accessType == READ_ACCESS_TYPE && itMap->second._satisfied);
					} else if (predecessorType == REDUCTION_ACCESS_TYPE) {
						assert(!predecessor->closesReduction());
						isSatisfied = false;
						
						if (accessType != REDUCTION_ACCESS_TYPE || predecessor->getReductionInfo() != access->getReductionInfo())
							predecessor->setClosesReduction(true);
					} else {
						isSatisfied = false;
					}
					
					// Reductions always start
					if (accessType == REDUCTION_ACCESS_TYPE)
						isSatisfied = true;
					
					predecessor->setSuccessor(task);
					
					itMap->second._satisfied = isSatisfied;
					
					if (!isSatisfied) task->increasePredecessors();
				} else {
					if (accessType == READ_ACCESS_TYPE || accessType == REDUCTION_ACCESS_TYPE)
						access->markAsTop();
					
					if (reductionInfo != nullptr && access->getReductionInfo() != reductionInfo) {
						closeReduction = reductionInfo->markAsClosed();
						assert(closeReduction);
					}
				}
			} // Lock Release
			
			if (closeReduction) {
				releaseReductionInfo(reductionInfo);
			}
			
			if (isSatisfied) {
				Instrument::dataAccessBecomesSatisfied(
						dataAccessInstrumentationId,
						true,
						task->getInstrumentationTaskId()
				);
			}
		}
	}
	
	void releaseReductionInfo(ReductionInfo *info)
	{
		assert(info != nullptr);
		assert(info != info->getOriginalAddress());
		assert(info->finished());
		
		__attribute__((unused)) bool wasLastCombination = info->combine(true);
		ObjectAllocator<ReductionInfo>::deleteObject(info);
		
		assert(wasLastCombination);
	}
	
	void decreaseDeletableCountOrDelete(Task *originator,
		CPUDependencyData::deletable_originator_list_t &deletableOriginators)
	{
		if (originator->getDataAccesses().decreaseDeletableCount() && originator->decreaseRemovalBlockingCount())
			deletableOriginators.push_back(originator); // Ensure destructor is called
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
	
	void combineTaskReductions(Task *task, ComputePlace *computePlace)
	{
		assert(task != nullptr);
		TaskDataAccesses &accessStruct = task->getDataAccesses();
		assert(!accessStruct.hasBeenDeleted());
		
		if (!accessStruct.hasDataAccesses()) return;
		
		for (size_t i = 0; i < accessStruct.getRealAccessNumber(); ++i) {
			DataAccess *access = &accessStruct._accessArray[i];
			if(access->getType() == REDUCTION_ACCESS_TYPE) {
				ReductionInfo *reductionInfo = access->getReductionInfo();
				reductionInfo->releaseSlotsInUse(((CPU *) computePlace)->getIndex());
			}
		}
	}
}

#pragma GCC visibility pop

