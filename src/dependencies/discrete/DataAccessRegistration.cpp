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
#include <numaif.h>

#include "BottomMapEntry.hpp"
#include "CommutativeSemaphore.hpp"
#include "CPUDependencyData.hpp"
#include "DataAccessRegistration.hpp"
#include "executors/threads/TaskFinalization.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/SpinWait.hpp"
#include "memory/manager-numa/ManagerNUMA.hpp"
#include "scheduling/Scheduler.hpp"
#include "TaskDataAccesses.hpp"
#include "tasks/Task.hpp"

#include <InstrumentDependenciesByAccessLinks.hpp>
#include <InstrumentDependencySubsystemEntryPoints.hpp>
#include <InstrumentTaskId.hpp>
#include <ObjectAllocator.hpp>

#define __unused __attribute__((unused))

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
		for (int i = 0; i < nanos6_device_t::nanos6_device_type_num; ++i) {
			ComputePlace *computePlaceHint = nullptr;
			if (computePlace != nullptr && computePlace->getType() == i)
				computePlaceHint = computePlace;

			ReadyTaskHint schedulingHint = SIBLING_TASK_HINT;
			if (fromBusyThread || !computePlaceHint || !computePlaceHint->isOwned()) {
				schedulingHint = BUSY_COMPUTE_PLACE_TASK_HINT;
			}

			CPUDependencyData::satisfied_originator_list_t &list = hpDependencyData.getSatisfiedOriginators(i);
			if (list.size() > 0) {
				Scheduler::addReadyTasks(
					(nanos6_device_t)i,
					list.getArray(),
					list.size(),
					computePlaceHint,
					schedulingHint);
			}
		}

		hpDependencyData.clearSatisfiedOriginators();

		for (Task *originator : hpDependencyData._satisfiedCommutativeOriginators) {
			ComputePlace *computePlaceHint = nullptr;
			if (computePlace != nullptr && originator->getDeviceType() == computePlace->getType())
				computePlaceHint = computePlace;

			ReadyTaskHint schedulingHint = SIBLING_TASK_HINT;
			if (fromBusyThread || !computePlaceHint || !computePlaceHint->isOwned()) {
				schedulingHint = BUSY_COMPUTE_PLACE_TASK_HINT;
			}

			Scheduler::addReadyTask(originator, computePlaceHint, schedulingHint);
		}

		hpDependencyData._satisfiedCommutativeOriginators.clear();
	}

	static inline void processDeletableOriginators(CPUDependencyData &hpDependencyData)
	{
		// As there is no "task garbage collection", the runtime will only destruct the tasks for us if we mark them as
		// not needed on the unregisterTaskDataAccesses call, so this takes care on tasks ended anywhere else.

		for (Task *deletableOriginator : hpDependencyData._deletableOriginators) {
			assert(deletableOriginator != nullptr);
			TaskFinalization::disposeTask(deletableOriginator);
		}

		hpDependencyData._deletableOriginators.clear();
	}

	static inline void satisfyTask(
		Task *task,
		CPUDependencyData &hpDependencyData,
		ComputePlace *computePlace,
		bool fromBusyThread)
	{
		if (task->decreasePredecessors()) {
			TaskDataAccesses &accessStruct = task->getDataAccesses();

			if (accessStruct._commutativeMask.any() && !CommutativeSemaphore::registerTask(task))
				return;

			if (task->getL2hint() != (unsigned int) -1 && !task->isTaskfor() && computePlace->getSuccessor() == nullptr) {
				computePlace->setSuccessor(task);
			} else {
				hpDependencyData.addSatisfiedOriginator(task, task->getDeviceType());
			}

			if (hpDependencyData.full())
				processSatisfiedOriginators(hpDependencyData, computePlace, fromBusyThread);
		}
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
		// Let's not allow combining reductions with other types as it causes problems.
		assert((oldType != REDUCTION_ACCESS_TYPE && newType != REDUCTION_ACCESS_TYPE) || (newType == oldType));

		access->setType(combineTypes(oldType, newType));

		// ! weak + weak = !weak :)
		if (access->isWeak() && !weak)
			access->setWeak(false);
	}

	void registerTaskDataAccess(
		Task *task, DataAccessType accessType, bool weak, void *address, size_t length,
		reduction_type_and_operator_index_t reductionTypeAndOperatorIndex,
		reduction_index_t reductionIndex, int symbolIndex)
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
		DataAccess *access = accessStruct.allocateAccess(address, accessType, task, length, weak, alreadyExisting);

		if (!alreadyExisting) {
			if (!weak) {
				accessStruct.incrementTotalDataSize(length);
			}

			if (accessType == REDUCTION_ACCESS_TYPE) {
				access->setReductionOperator(reductionTypeAndOperatorIndex);
				access->setReductionIndex(reductionIndex);
			}
		} else {
			upgradeAccess(access, accessType, weak);
		}

		access->addToSymbol(symbolIndex);

		// Tuning the number of deps of child taskloops
		task->increaseMaxChildDependencies();
	}

	void propagateMessages(
			CPUDependencyData &hpDependencyData,
			mailbox_t &mailBox,
			ReductionInfo *originalReductionInfo,
			ComputePlace *computePlace,
			bool fromBusyThread)
	{
		DataAccessMessage next;

		while (!mailBox.empty()) {
			next = mailBox.top();
			mailBox.pop();

			assert(next.from != nullptr);

			if (next.location) {
				if (next.to != nullptr) {
					DataTrackingSupport::DataTrackingInfo *trackingInfo = next.from->getTrackingInfo();
					DataTrackingSupport::location_t location = (trackingInfo != nullptr) ? trackingInfo->_location : DataTrackingSupport::UNKNOWN_LOCATION;
					if (location != DataTrackingSupport::UNKNOWN_LOCATION) {
						DataTrackingSupport::timestamp_t timeL2 = trackingInfo->_timeL2;
						DataTrackingSupport::timestamp_t timeL3 = trackingInfo->_timeL3;
						next.to->updateTrackingInfo(location, timeL2, timeL3);
					}

					next.to->setHomeNode(next.from->getHomeNode());
				}
			}

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
				// If this access represents a big share of the total data size, we can
				// directly set the L2 hint. We do not do the same for L3 because there
				// may be L2 locality with other L2, but if we set an L3 hint, it won't
				// be computed.
				if (DataTrackingSupport::isTrackingEnabled()) {
					double score = (double) next.from->getAccessRegion().getSize() / task->getDataAccesses().getTotalDataSize();
					if (score >= DataTrackingSupport::L2_THRESHOLD) {
						unsigned int &L2hint = task->getL2hint();
						unsigned int &L3hint = task->getL3hint();
						L2hint = ((CPU *)computePlace)->getL2CacheId();
						L3hint = ((CPU *)computePlace)->getL3CacheId();
					}
				}
				satisfyTask(task, hpDependencyData, computePlace, fromBusyThread);
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

	void finalizeDataAccess(
		Task *task,
		DataAccess *access,
		void *address,
		CPUDependencyData &hpDependencyData,
		ComputePlace *computePlace,
		bool fromBusyThread)
	{
		DataAccessType originalAccessType = access->getType();
		// No race, the parent is finished so all childs must be registered by now.
		DataAccess *childAccess = access->getChild();
		ReductionInfo *reductionInfo = nullptr;

		mailbox_t &mailBox = hpDependencyData._mailBox;
		assert(mailBox.empty());

		access_flags_t flagsToSet = ACCESS_UNREGISTERED;

		if (childAccess == nullptr) {
			flagsToSet |= (ACCESS_CHILD_WRITE_DONE | ACCESS_CHILD_READ_DONE | ACCESS_CHILD_CONCURRENT_DONE | ACCESS_CHILD_COMMUTATIVE_DONE);
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
			propagateMessages(hpDependencyData, mailBox, reductionInfo, computePlace, fromBusyThread);
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

		Instrument::enterRegisterTaskDataAcesses();

#ifndef NDEBUG
		{
			bool alreadyTaken = false;
			assert(hpDependencyData._inUse.compare_exchange_strong(alreadyTaken, true));
		}
#endif

		task->increasePredecessors(2);

		// This part creates the DataAccesses and inserts it to dependency system
		task->registerDependencies(/* discrete */ true);

		TaskDataAccesses &accessStructures = task->getDataAccesses();

		insertAccesses(task, hpDependencyData);

		assert(!accessStructures.hasBeenDeleted());
		if (accessStructures.hasDataAccesses()) {
			task->increaseRemovalBlockingCount();
		}

		processSatisfiedOriginators(hpDependencyData, computePlace, true);
		processDeletableOriginators(hpDependencyData);

#ifndef NDEBUG
		{
			bool alreadyTaken = true;
			assert(hpDependencyData._inUse.compare_exchange_strong(alreadyTaken, false));
		}
#endif

		bool ready = task->decreasePredecessors(2);

		// Commutative accesses have to acquire the commutative region
		if (ready && accessStructures._commutativeMask.any())
			ready = CommutativeSemaphore::registerTask(task);

		Instrument::exitRegisterTaskDataAcesses();

		return ready;
	}

	void unregisterTaskDataAccesses(Task *task, ComputePlace *computePlace, CPUDependencyData &hpDependencyData,
		__attribute__((unused)) MemoryPlace *location, bool fromBusyThread)
	{
		assert(task != nullptr);

		Instrument::enterUnregisterTaskDataAcesses();

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
			accessStruct.forAll([&](void *address, DataAccess *access) -> bool {
				// Skip if released
				if (!access->isReleased())
					finalizeDataAccess(task, access, address, hpDependencyData, computePlace, fromBusyThread);
				return true;
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
				// We cannot close this in case we had a weak reduction
				DataAccess *parentAccess = accessStruct.findAccess(itMap->first);

				if (parentAccess == nullptr || parentAccess->getType() != REDUCTION_ACCESS_TYPE) {
					assert(!reductionInfo->finished());
					if (reductionInfo->markAsClosed())
						releaseReductionInfo(reductionInfo);

					itMap->second._reductionInfo = nullptr;
				} else {
					assert(parentAccess->isWeak());
				}
			}
		}

		// Release commutative mask. The order is important, as this will add satisfied originators
		if (accessStruct._commutativeMask.any())
			CommutativeSemaphore::releaseTask(task, hpDependencyData);

		if (accessStruct.hasDataAccesses()) {
			// All TaskDataAccesses have a deletableCount of 1 for default, so this will return true unless
			// some read/reduction accesses have increased this as well because the task cannot be deleted yet.
			// It also plays an important role in ensuring that a task will not be deleted by another one while
			// it's performing the dependency release

			if (accessStruct.decreaseDeletableCount())
				task->decreaseRemovalBlockingCount();
		}

		processSatisfiedOriginators(hpDependencyData, computePlace, fromBusyThread);
		processDeletableOriginators(hpDependencyData);

#ifndef NDEBUG
		{
			bool alreadyTaken = true;
			assert(hpDependencyData._inUse.compare_exchange_strong(alreadyTaken, false));
		}
#endif

		Instrument::exitUnregisterTaskDataAcesses();
	}

	void handleEnterTaskwait(Task *task, ComputePlace *, CPUDependencyData &)
	{
		assert(task != nullptr);

		TaskDataAccesses &accessStruct = task->getDataAccesses();
		assert(!accessStruct.hasBeenDeleted());

		bottom_map_t &bottomMap = accessStruct._subaccessBottomMap;

		for (bottom_map_t::iterator itMap = bottomMap.begin(); itMap != bottomMap.end(); itMap++) {
			ReductionInfo *reductionInfo = itMap->second._reductionInfo;

			if (reductionInfo != nullptr) {
				DataAccess *parentAccess = accessStruct.findAccess(itMap->first);

				if (parentAccess == nullptr || parentAccess->getType() != REDUCTION_ACCESS_TYPE) {
					assert(!reductionInfo->finished());
					if (reductionInfo->markAsClosed())
						releaseReductionInfo(reductionInfo);

					itMap->second._reductionInfo = nullptr;
				} else {
					assert(parentAccess->isWeak());
				}
			}
		}
	}

	void handleExitTaskwait(Task *, ComputePlace *, CPUDependencyData &)
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
		accessStruct.forAll([&](void *address, DataAccess *access) -> bool {
			DataAccessType accessType = access->getType();
			ReductionInfo *reductionInfo = nullptr;
			DataAccess *predecessor = nullptr;
			bottom_map_t::iterator itMap;
			bool weak = access->isWeak();
			bool setHomeNode = true;

			// Instrumentation needs a region.
			Instrument::data_access_id_t dataAccessInstrumentationId = Instrument::createdDataAccess(
				nullptr, accessType, false, access->getAccessRegion(), false, false,
				false, Instrument::access_object_type_t::regular_access_type,
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

			if (accessType == COMMUTATIVE_ACCESS_TYPE && !weak) {
				// Calculate commutative mask
				CommutativeSemaphore::combineMaskAndAddress(accessStruct._commutativeMask, address);
			}

			bool dispose = false;
			bool schedule = false;
			DataAccessMessage fromCurrent;
			DataAccess *parentAccess = nullptr;

			if (predecessor == nullptr) {
				parentAccess = parentAccessStruct.findAccess(address);

				if (parentAccess != nullptr) {
					// In case we need to inherit reduction
					reductionInfo = parentAccess->getReductionInfo();
					// Check that if we got something the parent is weakreduction
					assert(reductionInfo == nullptr || parentAccess->isWeak());
				}
			}

			// Check if we're closing a reduction, or allocate one in case we need it.
			if (accessType == REDUCTION_ACCESS_TYPE) {
				// Get the reduction info from the bottom map. If there is none, check
				// if our parent has one (for weak reductions)
				ReductionInfo *currentReductionInfo = itMap->second._reductionInfo;
				reduction_type_and_operator_index_t typeAndOpIndex = access->getReductionOperator();
				size_t length = access->getLength();

				if (currentReductionInfo == nullptr) {
					currentReductionInfo = reductionInfo;
					// Inherited reductions must be equal
					assert(reductionInfo == nullptr || (reductionInfo->getTypeAndOperatorIndex() == typeAndOpIndex && reductionInfo->getOriginalLength() == length));
				} else {
					reductionInfo = currentReductionInfo;
				}

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

			if (predecessor == nullptr) {
				if (parentAccess != nullptr) {
					parentAccess->setChild(access);
					DataAccessMessage message = parentAccess->applySingle(ACCESS_HASCHILD, mailBox);
					fromCurrent = access->applySingle(message.flagsForNext, mailBox);
					schedule = fromCurrent.schedule;
					assert(!(fromCurrent.flagsForNext));

					dispose = parentAccess->applyPropagated(message);
					assert(!dispose);

					DataTrackingSupport::DataTrackingInfo *trackingInfo = parentAccess->getTrackingInfo();
					DataTrackingSupport::location_t location = (trackingInfo != nullptr) ? trackingInfo->_location : DataTrackingSupport::UNKNOWN_LOCATION;
					if (location != DataTrackingSupport::UNKNOWN_LOCATION) {
						DataTrackingSupport::timestamp_t timeL2 = trackingInfo->_timeL2;
						DataTrackingSupport::timestamp_t timeL3 = trackingInfo->_timeL3;
						access->updateTrackingInfo(location, timeL2, timeL3);
					}

					access->setHomeNode(parentAccess->getHomeNode());
					setHomeNode = false;
					if (dispose)
						decreaseDeletableCountOrDelete(parentTask, hpDependencyData._deletableOriginators);
				} else {
					schedule = true;
					fromCurrent = access->applySingle(
						ACCESS_READ_SATISFIED | ACCESS_WRITE_SATISFIED | ACCESS_CONCURRENT_SATISFIED | ACCESS_COMMUTATIVE_SATISFIED,
						mailBox);
				}
			} else {
				predecessor->setSuccessor(access);
				DataAccessMessage message = predecessor->applySingle(ACCESS_HASNEXT, mailBox);
				fromCurrent = access->applySingle(message.flagsForNext, mailBox);
				schedule = fromCurrent.schedule;
				assert(!(fromCurrent.flagsForNext));

				DataTrackingSupport::DataTrackingInfo *trackingInfo = predecessor->getTrackingInfo();
				DataTrackingSupport::location_t location = (trackingInfo != nullptr) ? trackingInfo->_location : DataTrackingSupport::UNKNOWN_LOCATION;
				if (location != DataTrackingSupport::UNKNOWN_LOCATION) {
					DataTrackingSupport::timestamp_t timeL2 = trackingInfo->_timeL2;
					DataTrackingSupport::timestamp_t timeL3 = trackingInfo->_timeL3;
					access->updateTrackingInfo(location, timeL2, timeL3);
				}
				access->setHomeNode(predecessor->getHomeNode());
				setHomeNode = false;

				dispose = predecessor->applyPropagated(message);
				if (dispose)
					decreaseDeletableCountOrDelete(predecessor->getOriginator(), hpDependencyData._deletableOriginators);
			}

			// The homenode couldn't be propagated, check it in the directory
			if (!weak && setHomeNode) {
				size_t length = access->getAccessRegion().getSize();
				uint8_t homenode = ManagerNUMA::getHomeNode(address, length);
				access->setHomeNode(homenode);
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

			return true; // Continue iteration
		});
	}

	static inline void releaseReductionInfo(ReductionInfo *info)
	{
		assert(info != nullptr);
		assert(info->finished());

		info->combine();
		ObjectAllocator<ReductionInfo>::deleteObject(info);
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
		__unused DataAccessType &dataAccessType, reduction_index_t reductionIndex,
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
		assert(computePlace != nullptr);
		assert(task->isRunnable());

		TaskDataAccesses &accessStruct = (task->isTaskfor() ? task->getParent()->getDataAccesses() : task->getDataAccesses());
		assert(!accessStruct.hasBeenDeleted());

		if (!accessStruct.hasDataAccesses())
			return;

		accessStruct.forAll([&](void *, DataAccess *access) -> bool {
			// Skip if released
			if (access->isReleased())
				return true;

			if (access->getType() == REDUCTION_ACCESS_TYPE && !access->isWeak()) {
				ReductionInfo *reductionInfo = access->getReductionInfo();
				reductionInfo->releaseSlotsInUse(task, computePlace);
			}

			return true;
		});
	}

	void translateReductionAddresses(Task *task, ComputePlace *computePlace,
		nanos6_address_translation_entry_t *translationTable,
		int totalSymbols)
	{
		assert(task != nullptr);
		assert(computePlace != nullptr);
		assert(translationTable != nullptr);

		// Initialize translationTable
		for (int i = 0; i < totalSymbols; ++i)
			translationTable[i] = {0, 0};

		TaskDataAccesses &accessStruct = task->getDataAccesses();

		assert(!accessStruct.hasBeenDeleted());

		accessStruct.forAll([&](void *address, DataAccess *access) {
			if (access->getType() == REDUCTION_ACCESS_TYPE && !access->isWeak()) {
				ReductionInfo *reductionInfo = access->getReductionInfo();
				assert(reductionInfo != nullptr);

				void *translation = reductionInfo->getFreeSlot(task, computePlace);

				for (int j = 0; j < totalSymbols; ++j) {
					if (access->isInSymbol(j)) {
						translationTable[j] = {(size_t)address, (size_t)translation};
					}
				}
			}

			return true; // Continue iteration
		});
	}

	void releaseAccessRegion(
		Task *task,
		void *address,
		DataAccessType accessType,
		bool weak,
		ComputePlace *computePlace,
		CPUDependencyData &hpDependencyData,
		__attribute__((unused)) MemoryPlace const *location)
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
			DataAccess *access = accessStruct.findAccess(address);

			// Some unlikely sanity checks
			FatalErrorHandler::failIf(access == nullptr,
				"Attempt to release an access that was not originally registered in the task");

			FatalErrorHandler::failIf(access->getType() != accessType || access->isWeak() != weak,
				"It is not possible to partially release a dependence.");

			// Release reduction storage before finalizing, as we might delete the ReductionInfo later
			if (access->getType() == REDUCTION_ACCESS_TYPE && !access->isWeak()) {
				ReductionInfo *reductionInfo = access->getReductionInfo();
				assert(reductionInfo != nullptr);
				reductionInfo->releaseSlotsInUse(task, computePlace);
			}

			finalizeDataAccess(task, access, address, hpDependencyData, computePlace, true);
		} else {
			FatalErrorHandler::fail("Attempt to release an access that was not originally registered in the task");
		}

		// Unfortunately, due to the CommutativeSemaphore implementation, we cannot release the commutative mask.
		// This is because it can be aliased between accesses, although if a counter was added for the number of
		// commutative accesses, it would be possible to find out how safe is it to release the mask.
		processSatisfiedOriginators(hpDependencyData, computePlace, true);
		processDeletableOriginators(hpDependencyData);

#ifndef NDEBUG
		{
			bool alreadyTaken = true;
			assert(hpDependencyData._inUse.compare_exchange_strong(alreadyTaken, false));
		}
#endif
	}

	void releaseTaskwaitFragment(
		__attribute__((unused)) Task *task,
		__attribute__((unused)) DataAccessRegion region,
		__attribute__((unused)) ComputePlace *computePlace,
		__attribute__((unused)) CPUDependencyData &hpDependencyData)
	{
		assert(false);
	}
} // namespace DataAccessRegistration

#pragma GCC visibility pop
