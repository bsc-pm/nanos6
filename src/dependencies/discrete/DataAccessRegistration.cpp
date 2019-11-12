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
#include "lowlevel/SpinWait.hpp"

#include <InstrumentDependenciesByAccessLinks.hpp>
#include <InstrumentTaskId.hpp>
#include <MemoryAllocator.hpp>
#include <ObjectAllocator.hpp>

#pragma GCC visibility push(hidden)

namespace DataAccessRegistration {
	typedef TaskDataAccesses::bottom_map_t bottom_map_t;

	static inline void insertAccesses(Task * task);

	static inline ReductionInfo * allocateReductionInfo(
		DataAccessType &dataAccessType, reduction_index_t reductionIndex,
		reduction_type_and_operator_index_t reductionTypeAndOpIndex,
		void * address, const size_t length, const Task &task);

	static inline void releaseReductionInfo(ReductionInfo *info);

	static inline void decreaseDeletableCountOrDelete(Task *originator,
		CPUDependencyData::deletable_originator_list_t &deletableOriginators);

	static inline bool matchAll(access_flags_t value, access_flags_t mask);

	//! Process all the originators that have become ready
	static inline void processSatisfiedOriginators(
		CPUDependencyData &hpDependencyData,
		ComputePlace *computePlace,
		bool fromBusyThread)
	{
		// The number of satisfied originators that will be added to the scheduler
		size_t numAddedTasks = hpDependencyData._satisfiedOriginators.size();

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

		if (numAddedTasks) {
			// After adding tasks, the CPUManager may want to unidle CPU(s)
			CPUManager::executeCPUManagerPolicy(computePlace, ADDED_TASKS, numAddedTasks);
		}

		// As there is no "task garbage collection", the runtime will only destruct the tasks for us if we mark them as
		// not needed on the unregisterTaskDataAccesses call, so this takes care on tasks ended anywhere else.

		for (Task *deletableOriginator : hpDependencyData._deletableOriginators) {
			assert(deletableOriginator != nullptr);
			TaskFinalization::disposeTask(deletableOriginator, computePlace);
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

	static inline void upgradeAccess(DataAccess * access, DataAccessType newType, bool weak)
	{
		DataAccessType oldType = access->getType();
		// Duping reductions is incorrect
		assert(oldType != REDUCTION_ACCESS_TYPE && newType != REDUCTION_ACCESS_TYPE);

		access->setType(combineTypes(oldType, newType));
		if(access->isWeak() && !weak)
			access->setWeak(false);
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

		TaskDataAccesses &accessStruct = task->getDataAccesses();

		assert(!accessStruct.hasBeenDeleted());

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
			upgradeAccess(alreadyExistingAccess, accessType, weak);
		}
	}


	bool shouldSchedule(Task * task, DataAccess * access, bool readSatisfied, bool writeSatisified, access_flags_t oldFlags) {
		// Weak access are already scheduled.
		if(access->isWeak())
			return false;
		else if(access->getType() == READ_ACCESS_TYPE && readSatisfied && !matchAll(oldFlags, ACCESS_READ_SATISFIED))
			return true;
		else if((access->getType() == WRITE_ACCESS_TYPE || access->getType() == READWRITE_ACCESS_TYPE) && writeSatisified && !matchAll(oldFlags, ACCESS_WRITE_SATISFIED))
			return true;
		else
			return false;
	}

	static inline bool checkBottomMap(Task * task, void * address, bool keepDeleting, DataAccess * access, bool isReduction) {
		assert(!task->getParent()->getDataAccesses().hasBeenDeleted());
		BottomMapEntry * entry = access->getBottomMapEntry();
		assert(entry != nullptr);

		if(keepDeleting)
			return entry->_access.compare_exchange_strong(access, nullptr);
		else
			return (entry->_access == access);
	}

	void satisfySuccessorFromChain(void * address, CPUDependencyData &hpDependencyData,
		bool keepDeleting, bool satisfyWrites, bool satisfyReads, Task * successor, DataAccess * currentAccessIterator,
		Task * currentTaskIterator, ReductionInfo * originalReductionInfo)
	{
		// NANOS6_VERB_OUT("Start: " << currentTaskIterator->getInstrumentationTaskId() << ":" << keepDeleting << satisfyWrites << satisfyReads);
		DataAccess * holdingOff = nullptr;
		Task * pendingToDelete = nullptr;

		while(keepDeleting || satisfyWrites || satisfyReads) {
			if(successor != nullptr) {
				if(successor == currentTaskIterator->getParent()) {
					// We are the last access in this chain. Travel to the parent (up one level).
					currentTaskIterator = successor;
					assert(!successor->getDataAccesses().hasBeenDeleted());
					currentAccessIterator = successor->getDataAccesses().findAccess(address);
					// This may be wrong and beyond useless.
					if(satisfyWrites)
						currentAccessIterator->setFlags(ACCESS_CHILDS_FINISHED);
					successor = currentAccessIterator->getSuccessor();
				} else {
					if(pendingToDelete != nullptr) {
						decreaseDeletableCountOrDelete(pendingToDelete, hpDependencyData._deletableOriginators);
						pendingToDelete = nullptr;
					}

					// NANOS6_VERB_OUT(successor->getInstrumentationTaskId() << ":" << keepDeleting << satisfyWrites << satisfyReads);
					// Propagate satisfiability and figure out next step
					// This is safe because successors don't change and if we're referencing it, it's not deleted.
					assert(!successor->getDataAccesses().hasBeenDeleted());
					DataAccess * nextAccess = successor->getDataAccesses().findAccess(address);
					assert(nextAccess != nullptr);
					assert((nextAccess->getFlags() & ACCESS_DELETABLE) == ACCESS_NONE);

					access_flags_t flagsToSet = ACCESS_NONE;

					if (keepDeleting)
						flagsToSet |= (ACCESS_DELETABLE | ACCESS_HOLDOFF);
					if(satisfyReads)
						flagsToSet |= ACCESS_READ_SATISFIED;
					if(satisfyWrites)
						flagsToSet |= ACCESS_WRITE_SATISFIED;

					access_flags_t oldFlags = nextAccess->setFlags(flagsToSet);

					if(shouldSchedule(successor, nextAccess, satisfyReads, satisfyWrites, oldFlags) && successor->decreasePredecessors())
						hpDependencyData._satisfiedOriginators.push_back(successor);

					if(satisfyWrites) {
						if(nextAccess->getType() == REDUCTION_ACCESS_TYPE) {
							ReductionInfo * reductionInfo = nextAccess->getReductionInfo();
							assert(reductionInfo != nullptr);

							if(!(oldFlags & ACCESS_WRITE_SATISFIED) && reductionInfo != originalReductionInfo) {
								if(reductionInfo->incrementUnregisteredAccesses())
									releaseReductionInfo(reductionInfo);

								originalReductionInfo = reductionInfo;
							}
						}

						satisfyWrites = ((oldFlags & ACCESS_UNREGISTERED)
								|| (nextAccess->isWeak() && nextAccess->getChild() != nullptr));
					}

					if (satisfyReads) {
						// This is not totally fine for read early releases.
						if(nextAccess->getType() == READ_ACCESS_TYPE && (!(oldFlags & ACCESS_READ_SATISFIED) || (oldFlags & ACCESS_UNREGISTERED)))
							satisfyReads = true;
						else if(nextAccess->isWeak() && (nextAccess->getChild() != nullptr || (oldFlags & ACCESS_UNREGISTERED)))
							satisfyReads = true;
						else if (oldFlags & ACCESS_UNREGISTERED)
							satisfyReads = true;
						else
							satisfyReads = false;
					}

					if(keepDeleting) {
						if(oldFlags & ACCESS_UNREGISTERED) {
							assert(pendingToDelete == nullptr);
							pendingToDelete = successor;
							nextAccess->unsetFlags(ACCESS_HOLDOFF);
						} else if(oldFlags & ACCESS_IN_TASKWAIT) {
							assert(nextAccess->getChild() != nullptr);
							nextAccess->unsetFlags(ACCESS_HOLDOFF);
						} else {
							assert(holdingOff == nullptr);
							holdingOff = nextAccess;
							keepDeleting = false;
						}
					}

					while((oldFlags & ACCESS_UNREGISTERED) && !(oldFlags & ACCESS_UNREGISTERING_DONE)) {
						spinWait();
						oldFlags = nextAccess->getFlags();
					}

					currentTaskIterator = successor;
					currentAccessIterator = nextAccess;

					if(nextAccess->getChild() != nullptr) {
						// Guaranteed because the child will register on the parent before inheriting the satisfied bits.
						successor = nextAccess->getChild();
					} else {
						successor = nextAccess->getSuccessor();
					}
				}
			} else {
				// Look at the bottom map.
				bool isReduction = currentAccessIterator->getType() == REDUCTION_ACCESS_TYPE;
				if(checkBottomMap(currentTaskIterator, address, keepDeleting, currentAccessIterator, isReduction)) {
					keepDeleting = satisfyReads = satisfyWrites = false;
				} else {
					successor = currentAccessIterator->getSuccessor();

					// This retry can happen multiple times.
					while(successor == nullptr) {
						spinWait();
						successor = currentAccessIterator->getSuccessor();
					}
				}
			}
		}

		if(holdingOff != nullptr)
			holdingOff->unsetFlags(ACCESS_HOLDOFF);

		if(pendingToDelete != nullptr)
			decreaseDeletableCountOrDelete(pendingToDelete, hpDependencyData._deletableOriginators);
	}

	void walkAccessChain(Task *task, DataAccess *access,
			void *address, CPUDependencyData &hpDependencyData) {
		DataAccess * currentAccessIterator = access;
		DataAccessType originalAccessType = access->getType();
		Task * successor = currentAccessIterator->getSuccessor();
		// No race, the parent is finished so all childs must be registered by now.
		Task * childTask = access->getChild();
		ReductionInfo *reductionInfo = nullptr;

		access_flags_t flagsToSet = ACCESS_UNREGISTERED;

		if(childTask == nullptr) {
			flagsToSet |= ACCESS_CHILDS_FINISHED;
		} else {
			// Place ourselves as successors of the last access.
			DataAccess * lastChild = nullptr;
			TaskDataAccesses &taskAccesses = task->getDataAccesses();
			assert(!taskAccesses.hasBeenDeleted());

			bottom_map_t &bottomMap = taskAccesses._subaccessBottomMap;
			bottom_map_t::iterator itMap = bottomMap.find(address);
			assert(itMap != bottomMap.end());
			BottomMapEntry &node = itMap->second;
			lastChild = node._access;

			// Maybe this doesn't need to be CAS and can be an assign.
			while (!node._access.compare_exchange_weak(lastChild, access));
			assert(lastChild != nullptr);
			lastChild->setSuccessor(task);
		}

		if(originalAccessType == REDUCTION_ACCESS_TYPE) {
			reductionInfo = access->getReductionInfo();
			assert(reductionInfo != nullptr);

			if(reductionInfo->incrementUnregisteredAccesses()) {
				releaseReductionInfo(reductionInfo);
			}
		}

		access_flags_t oldFlags = access->setFlags(flagsToSet);
		bool keepDeleting = (oldFlags & ACCESS_DELETABLE);
		bool satisfyReads = (oldFlags & ACCESS_READ_SATISFIED);
		bool satisfyWrites = (oldFlags & ACCESS_WRITE_SATISFIED);

		while(oldFlags & ACCESS_HOLDOFF) {
			spinWait();
			oldFlags = access->getFlags();
		}

		if(originalAccessType == READ_ACCESS_TYPE)
			satisfyReads = satisfyWrites;

		if(keepDeleting)
			decreaseDeletableCountOrDelete(task, hpDependencyData._deletableOriginators);

		if(childTask != nullptr) {
			satisfySuccessorFromChain(address, hpDependencyData, keepDeleting, satisfyWrites, satisfyReads, childTask,
				access, task, reductionInfo);
		} else {
			satisfySuccessorFromChain(address, hpDependencyData, keepDeleting, satisfyWrites, satisfyReads, successor,
				access, task, reductionInfo);
		}

		access->setFlags(ACCESS_UNREGISTERING_DONE);
	}

	void finalizeDataAccess(Task *task, DataAccess *access, void *address,
		CPUDependencyData &hpDependencyData, ComputePlace *computePlace)
	{
		assert(computePlace != nullptr);

		walkAccessChain(task, access, address, hpDependencyData);
	}

	bool registerTaskDataAccesses(Task *task, __attribute__((unused)) ComputePlace *computePlace, __attribute__((unused)) CPUDependencyData &hpDependencyData)
	{
		// This is called once per task, and will create all the dependencies in register_depinfo, to later insert
		// them into the chain in the insertAccesses call.

		assert(task != nullptr);
		assert(computePlace != nullptr);

		nanos6_task_info_t *taskInfo = task->getTaskInfo();
		assert(taskInfo != 0);

		task->increasePredecessors(2);

		// This part creates the DataAccesses and inserts it to dependency system
		taskInfo->register_depinfo(task->getArgsBlock(), nullptr, task);

		TaskDataAccesses &accessStructures = task->getDataAccesses();

		insertAccesses(task);

		assert(!accessStructures.hasBeenDeleted());
		if (accessStructures.hasDataAccesses()) {
			task->increaseRemovalBlockingCount();
		}

		return task->decreasePredecessors(2);
	}

	void unregisterTaskDataAccesses(Task *task, ComputePlace *computePlace, CPUDependencyData &hpDependencyData,
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
			// It also plays an important role in ensuring that a task will not be deleted by another one while
			// it's performing the dependency release

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

	void handleEnterBlocking(__attribute__((unused)) Task *task)
	{
	}

	void handleExitBlocking(__attribute__((unused)) Task *task)
	{
	}

	static inline void unlockAccessInTaskwait(Task *task, DataAccess * access, void * address, CPUDependencyData &hpDependencyData) {
		assert(task != nullptr);
		assert(access != nullptr);

		access_flags_t oldFlags = access->setFlags(ACCESS_IN_TASKWAIT);
		assert(!(oldFlags & ACCESS_IN_TASKWAIT));

		if(oldFlags & ACCESS_DELETABLE) {
			Task * child = access->getChild();
			assert(child != nullptr);
			satisfySuccessorFromChain(address, hpDependencyData, true, false, false, child,
				access, task, nullptr);
			access->setChild(nullptr);
		}
	}

	void handleEnterTaskwait(Task *task, ComputePlace *computePlace, CPUDependencyData &dependencyData)
	{
		assert(task != nullptr);

		TaskDataAccesses &accessStruct = task->getDataAccesses();
		assert(!accessStruct.hasBeenDeleted());

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

		DataAccess * accessArray = accessStruct._accessArray;
		void ** addressArray = accessStruct._addressArray;

		// We need to "unlock" our child accesses
		for (size_t i = 0; i < accessStruct.getRealAccessNumber(); ++i) {
			DataAccess * access = &accessArray[i];
			if(access->getChild() != nullptr)
				unlockAccessInTaskwait(task, access, addressArray[i], dependencyData);
		}

		processSatisfiedOriginators(dependencyData, computePlace, true);
	}

	void handleExitTaskwait(__attribute__((unused)) Task *task, __attribute__((unused)) ComputePlace *computePlace,
		__attribute__((unused)) CPUDependencyData &dependencyData)
	{
		TaskDataAccesses &accessStruct = task->getDataAccesses();
		assert(!accessStruct.hasBeenDeleted());

		DataAccess * accessArray = accessStruct._accessArray;

		// Remove taskwait flags
		for (size_t i = 0; i < accessStruct.getRealAccessNumber(); ++i) {
			DataAccess * access = &accessArray[i];
			if(access->getFlags() & ACCESS_IN_TASKWAIT)
				access->unsetFlags(ACCESS_IN_TASKWAIT);
		}
	}

	void handleTaskRemoval(__attribute__((unused)) Task *task, __attribute__((unused)) ComputePlace *computePlace)
	{
	}
		// Only needed for weak accesses (not implemented)
	static inline bool matchAll(access_flags_t value, access_flags_t mask) {
		return ((value & mask) == mask);
	}

	static inline bool setFlagsToAccess(DataAccess * current, access_flags_t flagsToSet) {
		access_flags_t oldFlags = current->setFlags(flagsToSet);

		if(current->getType() == READ_ACCESS_TYPE) {
			// We schedule if we satisfied the access (if oldFlags was already set, we raced and lost).
			return (flagsToSet & ACCESS_READ_SATISFIED) && !(oldFlags & ACCESS_READ_SATISFIED);
		} else {
			return (flagsToSet & ACCESS_WRITE_SATISFIED) && !(oldFlags & ACCESS_WRITE_SATISFIED);
		}
	}

	static inline bool inheritFromParent(DataAccess * parent, DataAccess * current) {
		if(parent == nullptr) {
			current->setFlags(ACCESS_WRITE_SATISFIED | ACCESS_READ_SATISFIED | ACCESS_DELETABLE);
			return true;
		}

		access_flags_t flagsParent = parent->getFlags();
		access_flags_t flagsToSet = flagsParent & (ACCESS_WRITE_SATISFIED | ACCESS_READ_SATISFIED);

		assert(!(flagsToSet & ACCESS_DELETABLE));

		return setFlagsToAccess(current, flagsToSet);
	}

	static inline bool inheritFromPredecessor(DataAccess * predecessor, DataAccess * current) {
		access_flags_t flagsPredecessor = predecessor->getFlags();
		access_flags_t flagsToSet = ACCESS_NONE;

		if(matchAll(flagsPredecessor, ACCESS_CHILDS_FINISHED | ACCESS_UNREGISTERED)) {
			// Predecessor has finished, inherit
			flagsToSet = flagsPredecessor & (ACCESS_WRITE_SATISFIED | ACCESS_READ_SATISFIED);
		} else if((flagsPredecessor & (ACCESS_READ_SATISFIED)) && predecessor->getType() == READ_ACCESS_TYPE) {
			flagsToSet = ACCESS_READ_SATISFIED;
		}

		return setFlagsToAccess(current, flagsToSet);
	}

	static inline void insertAccesses(Task *task)
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
			bool weak = access->isWeak();
			bool closeReduction = false;
			bottom_map_t::iterator itMap;
			ReductionInfo *reductionInfo = nullptr;
			DataAccess *predecessor = nullptr;

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
			// Determine our predecessor safely, and maybe insert ourselves to the map.
			std::pair<bottom_map_t::iterator, bool> result = addresses.emplace(std::piecewise_construct,
				std::forward_as_tuple(address),
				std::forward_as_tuple(access));

			itMap = result.first;
			access->setBottomMapEntry(&itMap->second);

			if(!result.second) {
				// Element already exists.
				predecessor = itMap->second._access;
				while(!itMap->second._access.compare_exchange_weak(predecessor, access))
					spinWait();
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

			// The "registration" was atomic because of the bottom map. Now we are outside of the lock section, and we
			// have to satisfy the access if needed in an atomical way.

			bool schedule = false;

			if(predecessor == nullptr) {
				DataAccess * parentAccess = parentAccessStruct.findAccess(address);

				// Important ordering: setChild _before_ inherit flags.
				// We can race against other accesses satisfying the parent.
				if(parentAccess != nullptr)
					parentAccess->setChild(task);

				schedule = inheritFromParent(parentAccess, access);

				if(reductionInfo != nullptr && access->getReductionInfo() != reductionInfo) {
					closeReduction = reductionInfo->markAsClosed();
					assert(closeReduction);
				}
			} else {
				// Important ordering: inherit _before_ setSuccessor.
				// The bottom map insertion is the point of syncronization between both accesses.
				schedule = inheritFromPredecessor(predecessor, access);
				predecessor->setSuccessor(task);

				if(predecessor->getType() == REDUCTION_ACCESS_TYPE &&
					(accessType != REDUCTION_ACCESS_TYPE || predecessor->getReductionInfo() != access->getReductionInfo())
				) {
					closeReduction = predecessor->setClosesReduction(true);
				}
			}

			// Weaks and reductions always start
			if(accessType == REDUCTION_ACCESS_TYPE || weak)
				schedule = true;

			if(!schedule)
				task->increasePredecessors();

			if(closeReduction)
				releaseReductionInfo(reductionInfo);

			if (schedule) {
				Instrument::dataAccessBecomesSatisfied(
						dataAccessInstrumentationId,
						true,
						task->getInstrumentationTaskId()
				);
			}
		}
	}

	static inline void releaseReductionInfo(ReductionInfo *info)
	{
		assert(info != nullptr);
		assert(info != info->getOriginalAddress());
		assert(info->finished());

		__attribute__((unused)) bool wasLastCombination = info->combine(true);
		ObjectAllocator<ReductionInfo>::deleteObject(info);

		assert(wasLastCombination);
	}

	static inline void decreaseDeletableCountOrDelete(Task *originator,
		CPUDependencyData::deletable_originator_list_t &deletableOriginators)
	{
		if (originator->getDataAccesses().decreaseDeletableCount()) {
			if(originator->decreaseRemovalBlockingCount()) {
				deletableOriginators.push_back(originator); // Ensure destructor is called
			}
		}
	}

	static inline ReductionInfo *allocateReductionInfo(
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

