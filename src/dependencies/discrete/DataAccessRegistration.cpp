/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
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
#include "lowlevel/SpinWait.hpp"
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

	static inline void insertAccesses(Task *task, CPUDependencyData &hpDependencyData);

	static inline ReductionInfo *allocateReductionInfo(
		DataAccessType &dataAccessType, reduction_index_t reductionIndex,
		reduction_type_and_operator_index_t reductionTypeAndOpIndex,
		void *address, const size_t length, const Task &task);

	static inline void releaseReductionInfo(ReductionInfo *info);

	static inline void decreaseDeletableCountOrDelete(Task *originator,
		CPUDependencyData::deletable_originator_list_t &deletableOriginators);

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
				(fromBusyThread ? BUSY_COMPUTE_PLACE_TASK_HINT : SIBLING_TASK_HINT));
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

	static inline void upgradeAccess(DataAccess *access, DataAccessType newType, bool weak)
	{
		DataAccessType oldType = access->getType();
		// Duping reductions is incorrect
		assert(oldType != REDUCTION_ACCESS_TYPE && newType != REDUCTION_ACCESS_TYPE);

		access->setType(combineTypes(oldType, newType));
		if (access->isWeak() && !weak)
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

		bool alreadyExisting;
		DataAccess *access = accessStruct.allocateAccess(address, accessType, task, weak, alreadyExisting);

		if (!alreadyExisting) {
			if (accessType == REDUCTION_ACCESS_TYPE) {
				access->setReductionOperator(reductionTypeAndOperatorIndex);
				access->setReductionLength(length);
				access->setReductionIndex(reductionIndex);
			}
		} else {
			upgradeAccess(access, accessType, weak);
		}
	}

	void propagateMessages(CPUDependencyData &hpDependencyData,
		mailbox_t &mailBox, ReductionInfo *originalReductionInfo)
	{
		DataAccessMessage next;

		while (!mailBox.empty()) {
			next = mailBox.top();
			mailBox.pop();

			assert(next.from != nullptr);

			if (next.to != nullptr && next.flagsForNext) {
				if (next.to->apply(next, mailBox)) {
					Task *task = next.to->getOriginator();
					assert(!task->getDataAccesses().hasBeenDeleted());
					assert(next.to != next.from);
					decreaseDeletableCountOrDelete(task, hpDependencyData._deletableOriginators);
				}
			}

			bool dispose = false;

			if (next.schedule) {
				assert(!next.from->getOriginator()->getDataAccesses().hasBeenDeleted());
				Task *task = next.from->getOriginator();
				assert(!task->getDataAccesses().hasBeenDeleted());
				if (task->decreasePredecessors())
					hpDependencyData._satisfiedOriginators.push_back(task);
			}

			if (next.combine) {
				assert(!next.from->getOriginator()->getDataAccesses().hasBeenDeleted());
				ReductionInfo *reductionInfo = next.from->getReductionInfo();
				assert(reductionInfo != nullptr);

				if (reductionInfo != originalReductionInfo) {
					if (reductionInfo->incrementUnregisteredAccesses())
						releaseReductionInfo(reductionInfo);
					originalReductionInfo = reductionInfo;
				}
			}

			if (next.flagsAfterPropagation) {
				assert(!next.from->getOriginator()->getDataAccesses().hasBeenDeleted());
				dispose = next.from->applyPropagated(next);
			}

			if (dispose) {
				Task *task = next.from->getOriginator();
				assert(!task->getDataAccesses().hasBeenDeleted());
				decreaseDeletableCountOrDelete(task, hpDependencyData._deletableOriginators);
			}
		}
	}

	void finalizeDataAccess(Task *task, DataAccess *access, void *address,
		CPUDependencyData &hpDependencyData)
	{
		DataAccessType originalAccessType = access->getType();
		// No race, the parent is finished so all childs must be registered by now.
		DataAccess *childAccess = access->getChild();
		ReductionInfo *reductionInfo = nullptr;

		mailbox_t &mailBox = hpDependencyData._mailBox;
		assert(mailBox.empty());

		access_flags_t flagsToSet = ACCESS_UNREGISTERED;

		if (childAccess == nullptr) {
			flagsToSet |= (ACCESS_CHILD_WRITE_DONE | ACCESS_CHILD_READ_DONE);
		} else {
			// Place ourselves as successors of the last access.
			DataAccess *lastChild = nullptr;
			TaskDataAccesses &taskAccesses = task->getDataAccesses();
			assert(!taskAccesses.hasBeenDeleted());

			bottom_map_t &bottomMap = taskAccesses._subaccessBottomMap;
			bottom_map_t::iterator itMap = bottomMap.find(address);
			assert(itMap != bottomMap.end());
			BottomMapEntry &node = itMap->second;
			lastChild = node._access;
			assert(lastChild != nullptr);

			lastChild->setSuccessor(access);
			DataAccessMessage m = lastChild->applySingle(ACCESS_HASNEXT | ACCESS_NEXTISPARENT, mailBox);
			__attribute__((unused)) DataAccessMessage m_debug = access->applySingle(m.flagsForNext, mailBox);
			assert(!(m_debug.flagsForNext));
			lastChild->applyPropagated(m);
		}

		if (originalAccessType == REDUCTION_ACCESS_TYPE) {
			reductionInfo = access->getReductionInfo();
			assert(reductionInfo != nullptr);

			if (reductionInfo->incrementUnregisteredAccesses()) {
				releaseReductionInfo(reductionInfo);
			}
		}

		// No need to worry here because no other thread can destroy this access, since we have to
		// finish unregistering all accesses before that can happen.
		DataAccessMessage message;
		message.to = message.from = access;
		message.flagsForNext = flagsToSet;
		bool dispose = access->apply(message, mailBox);

		if (!mailBox.empty()) {
			propagateMessages(hpDependencyData, mailBox, reductionInfo);
			assert(!dispose);
		} else if (dispose) {
			decreaseDeletableCountOrDelete(task, hpDependencyData._deletableOriginators);
		}
	}

	bool registerTaskDataAccesses(Task *task, ComputePlace *computePlace, CPUDependencyData &hpDependencyData)
	{
		// This is called once per task, and will create all the dependencies in register_depinfo, to later insert
		// them into the chain in the insertAccesses call.

		assert(task != nullptr);
		assert(computePlace != nullptr);

#ifndef NDEBUG
		{
			bool alreadyTaken = false;
			assert(hpDependencyData._inUse.compare_exchange_strong(alreadyTaken, true));
		}
#endif

		nanos6_task_info_t *taskInfo = task->getTaskInfo();
		assert(taskInfo != 0);

		task->increasePredecessors(2);

		// This part creates the DataAccesses and inserts it to dependency system
		taskInfo->register_depinfo(task->getArgsBlock(), nullptr, task);

		TaskDataAccesses &accessStructures = task->getDataAccesses();

		insertAccesses(task, hpDependencyData);

		assert(!accessStructures.hasBeenDeleted());
		if (accessStructures.hasDataAccesses()) {
			task->increaseRemovalBlockingCount();
		}

		processSatisfiedOriginators(hpDependencyData, computePlace, true);

#ifndef NDEBUG
		{
			bool alreadyTaken = true;
			assert(hpDependencyData._inUse.compare_exchange_strong(alreadyTaken, false));
		}
#endif

		return task->decreasePredecessors(2);
	}

	void unregisterTaskDataAccesses(Task *task, ComputePlace *computePlace, CPUDependencyData &hpDependencyData,
		__attribute__((unused)) MemoryPlace *location, bool fromBusyThread)
	{
		assert(task != nullptr);

		TaskDataAccesses &accessStruct = task->getDataAccesses();
		assert(!accessStruct.hasBeenDeleted());
		assert(hpDependencyData._mailBox.empty());

#ifndef NDEBUG
		{
			bool alreadyTaken = false;
			assert(hpDependencyData._inUse.compare_exchange_strong(alreadyTaken, true));
		}
#endif

		if (accessStruct.hasDataAccesses()) {
			// Release dependencies of all my accesses
			accessStruct.forAll([&](void *address, DataAccess *access) {
				finalizeDataAccess(task, access, address, hpDependencyData);
			});
		}

		bottom_map_t &bottomMap = accessStruct._subaccessBottomMap;

		for (bottom_map_t::iterator itMap = bottomMap.begin(); itMap != bottomMap.end(); itMap++) {
			DataAccess *access = itMap->second._access;
			assert(access != nullptr);
			DataAccessMessage m;
			m.from = m.to = access;
			m.flagsAfterPropagation = ACCESS_PARENT_DONE;
			if (access->applyPropagated(m))
				decreaseDeletableCountOrDelete(access->getOriginator(), hpDependencyData._deletableOriginators);

			ReductionInfo *reductionInfo = itMap->second._reductionInfo;
			if (reductionInfo != nullptr) {
				assert(!reductionInfo->finished());
				if (reductionInfo->markAsClosed())
					releaseReductionInfo(reductionInfo);

				itMap->second._reductionInfo = nullptr;
			}
		}

		if (accessStruct.hasDataAccesses()) {
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

	void handleEnterTaskwait(Task *task, __attribute__((unused)) ComputePlace *computePlace,
		__attribute__((unused)) CPUDependencyData &dependencyData)
	{
		assert(task != nullptr);

		TaskDataAccesses &accessStruct = task->getDataAccesses();
		assert(!accessStruct.hasBeenDeleted());

		bottom_map_t &bottomMap = accessStruct._subaccessBottomMap;

		for (bottom_map_t::iterator itMap = bottomMap.begin(); itMap != bottomMap.end(); itMap++) {
			ReductionInfo *reductionInfo = itMap->second._reductionInfo;

			if (reductionInfo != nullptr) {
				assert(!reductionInfo->finished());
				if (reductionInfo->markAsClosed())
					releaseReductionInfo(reductionInfo);

				itMap->second._reductionInfo = nullptr;
			}
		}
	}

	void handleExitTaskwait(__attribute__((unused)) Task *task, __attribute__((unused)) ComputePlace *computePlace,
		__attribute__((unused)) CPUDependencyData &dependencyData)
	{
	}

	void handleTaskRemoval(__attribute__((unused)) Task *task, __attribute__((unused)) ComputePlace *computePlace)
	{
	}

	static inline void insertAccesses(Task *task, CPUDependencyData &hpDependencyData)
	{
		TaskDataAccesses &accessStruct = task->getDataAccesses();
		assert(!accessStruct.hasBeenDeleted());

		Task *parentTask = task->getParent();
		assert(parentTask != nullptr);

		TaskDataAccesses &parentAccessStruct = parentTask->getDataAccesses();
		assert(!parentAccessStruct.hasBeenDeleted());

		mailbox_t &mailBox = hpDependencyData._mailBox;
		assert(mailBox.empty());

		// Default deletableCount of 1.
		accessStruct.increaseDeletableCount();

		// Get all seqs
		accessStruct.forAll([&](void *address, DataAccess *access) {
			DataAccessType accessType = access->getType();
			ReductionInfo *reductionInfo = nullptr;
			DataAccess *predecessor = nullptr;
			bottom_map_t::iterator itMap;
			bool weak = access->isWeak();

			// Instrumentation mock(for now)
			DataAccessRegion mock(address, 1);
			Instrument::data_access_id_t dataAccessInstrumentationId = Instrument::createdDataAccess(
				nullptr,
				accessType, false, mock,
				false, false, false, Instrument::access_object_type_t::regular_access_type,
				task->getInstrumentationTaskId());

			accessStruct.increaseDeletableCount();
			access->setInstrumentationId(dataAccessInstrumentationId);

			bottom_map_t &addresses = parentAccessStruct._subaccessBottomMap;
			// Determine our predecessor safely, and maybe insert ourselves to the map.
			std::pair<bottom_map_t::iterator, bool> result = addresses.emplace(std::piecewise_construct,
				std::forward_as_tuple(address),
				std::forward_as_tuple(access));

			itMap = result.first;

			if (!result.second) {
				// Element already exists.
				predecessor = itMap->second._access;
				itMap->second._access = access;
			}

			// Check if we're closing a reduction, or allocate one in case we need it.
			if (accessType == REDUCTION_ACCESS_TYPE) {
				ReductionInfo *currentReductionInfo = itMap->second._reductionInfo;
				reductionInfo = currentReductionInfo;
				reduction_type_and_operator_index_t typeAndOpIndex = access->getReductionOperator();
				size_t length = access->getReductionLength();

				if (currentReductionInfo == nullptr || currentReductionInfo->getTypeAndOperatorIndex() != typeAndOpIndex || currentReductionInfo->getOriginalLength() != length) {
					currentReductionInfo = allocateReductionInfo(accessType, access->getReductionIndex(), typeAndOpIndex,
						address, length, *task);
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

			bool schedule = false;
			bool dispose = false;
			DataAccessMessage fromCurrent;

			if (predecessor == nullptr) {
				DataAccess *parentAccess = parentAccessStruct.findAccess(address);

				if (parentAccess != nullptr) {
					parentAccess->setChild(access);
					DataAccessMessage message = parentAccess->applySingle(ACCESS_HASCHILD, mailBox);
					fromCurrent = access->applySingle(message.flagsForNext, mailBox);
					schedule = fromCurrent.schedule;
					assert(!(fromCurrent.flagsForNext));

					dispose = parentAccess->applyPropagated(message);
					assert(!dispose);
					if (dispose)
						decreaseDeletableCountOrDelete(parentTask, hpDependencyData._deletableOriginators);
				} else {
					schedule = true;
					fromCurrent = access->applySingle(ACCESS_READ_SATISFIED | ACCESS_WRITE_SATISFIED, mailBox);
				}
			} else {
				predecessor->setSuccessor(access);
				DataAccessMessage message = predecessor->applySingle(ACCESS_HASNEXT, mailBox);
				fromCurrent = access->applySingle(message.flagsForNext, mailBox);
				schedule = fromCurrent.schedule;
				assert(!(fromCurrent.flagsForNext));

				dispose = predecessor->applyPropagated(message);
				if (dispose)
					decreaseDeletableCountOrDelete(predecessor->getOriginator(), hpDependencyData._deletableOriginators);
			}

			if (fromCurrent.combine) {
				assert(access->getType() == REDUCTION_ACCESS_TYPE);
				assert(fromCurrent.flagsAfterPropagation == ACCESS_REDUCTION_COMBINED);
				ReductionInfo *current = access->getReductionInfo();
				if (current != reductionInfo) {
					dispose = current->incrementUnregisteredAccesses();
					assert(!dispose);
				}

				dispose = access->applyPropagated(fromCurrent);
				assert(!dispose);
			}

			if (reductionInfo != nullptr && access->getReductionInfo() != reductionInfo) {
				if (reductionInfo->markAsClosed())
					releaseReductionInfo(reductionInfo);
			}

			// Weaks and reductions always start
			if (accessType == REDUCTION_ACCESS_TYPE || weak)
				schedule = true;

			if (!schedule)
				task->increasePredecessors();

			if (schedule) {
				Instrument::dataAccessBecomesSatisfied(
					dataAccessInstrumentationId,
					true,
					task->getInstrumentationTaskId());
			}
		});
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
			if (originator->decreaseRemovalBlockingCount()) {
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

		if (!accessStruct.hasDataAccesses())
			return;

		accessStruct.forAll([&](void *, DataAccess *access) {
			if (access->getType() == REDUCTION_ACCESS_TYPE) {
				ReductionInfo *reductionInfo = access->getReductionInfo();
				reductionInfo->releaseSlotsInUse(((CPU *)computePlace)->getIndex());
			}
		});
	}

	void releaseAccessRegion(
		__attribute__((unused)) Task *task,
		__attribute__((unused)) void *address,
		__attribute__((unused)) DataAccessType accessType,
		__attribute__((unused)) bool weak,
		__attribute__((unused)) ComputePlace *computePlace,
		__attribute__((unused)) CPUDependencyData &hpDependencyData,
		__attribute__((unused)) MemoryPlace const *location)
	{

	}
} // namespace DataAccessRegistration

#pragma GCC visibility pop
