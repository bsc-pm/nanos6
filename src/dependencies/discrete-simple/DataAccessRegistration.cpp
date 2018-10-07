/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <cassert>
#include <deque>
#include <mutex>

#include "CPUDependencyData.hpp"
#include "DataAccess.hpp"
#include "DataAccessRegistration.hpp"
#include "DataAccessSequenceImplementation.hpp"
#include <MemoryAllocator.hpp>

#include "executors/threads/TaskFinalization.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"

#include <iostream>

#pragma GCC visibility push(hidden)

namespace DataAccessRegistration {
	typedef TaskDataAccesses::addresses_t addresses_t;
	typedef TaskDataAccesses::address_list_t address_list_t;
	
	
	//! Process all the originators that have become ready
	static inline void processSatisfiedOriginators(
		/* INOUT */ CPUDependencyData &hpDependencyData,
		ComputePlace *computePlace,
		bool fromBusyThread
	) {
		// NOTE: This is done without the lock held and may be slow since it can enter the scheduler
		for (Task *satisfiedOriginator : hpDependencyData._satisfiedOriginators) {
			assert(satisfiedOriginator != 0);
			
			ComputePlace *computePlaceHint = nullptr;
			if (computePlace != nullptr) {
				if (computePlace->getType() == satisfiedOriginator->getDeviceType()) {
					computePlaceHint = computePlace;
				}
			}
			
			ComputePlace *idleComputePlace = Scheduler::addReadyTask(
				satisfiedOriginator,
				computePlaceHint,
				(fromBusyThread ?
					SchedulerInterface::SchedulerInterface::BUSY_COMPUTE_PLACE_TASK_HINT
					: SchedulerInterface::SchedulerInterface::SIBLING_TASK_HINT
				)
			);
			
			if (idleComputePlace != nullptr) {
				ThreadManager::resumeIdle((CPU *) idleComputePlace);
			}
		}
		
		hpDependencyData._satisfiedOriginators.clear();
	}
	
	void registerTaskDataAccess(Task *task, DataAccessType accessType, void *address) {
		assert(task != nullptr);
		assert(address != nullptr);
		
		TaskDataAccesses &accessStruct = task->getDataAccesses();
		assert(!accessStruct.hasBeenDeleted());
		
		Task *parentTask = task->getParent();
		assert(parentTask != nullptr);
		
		TaskDataAccesses &parentAccessStruct = parentTask->getDataAccesses();
		assert(!parentAccessStruct.hasBeenDeleted());
		addresses_t &addresses = parentAccessStruct._dataAccessSequences;
		
		DataAccessSequence *sequence = nullptr;
		{
			std::lock_guard<TaskDataAccesses::spinlock_t> guard(parentAccessStruct._lock);
			
			addresses_t::iterator it = addresses.find(address);
			if (it != addresses.end()) {
				sequence = it->second;
				assert(sequence != nullptr);
			} else {
				sequence = MemoryAllocator::newObject<DataAccessSequence>();
				assert(sequence != nullptr);
				
				std::pair<void *, DataAccessSequence *> entry(address, sequence);
				addresses.insert(entry);
			}
		}
		
		
		address_list_t &reads = accessStruct._readAccessAddresses;
		address_list_t &writes = accessStruct._writeAccessAddresses;
		
		bool upgraded = false;
		bool registered = false;
		bool becameUnsatisfied = false;
		{
			std::lock_guard<DataAccessSequence::spinlock_t> guard(sequence->_lock);
			
			if (sequence->registeredLastDataAccess(task)) {
				if (accessType != READ_ACCESS_TYPE) {
					DataAccessType prevAccessType;
					becameUnsatisfied = sequence->upgradeLastDataAccess(&prevAccessType);
					upgraded = (prevAccessType == READ_ACCESS_TYPE);
				}
			} else {
				becameUnsatisfied = !sequence->registerDataAccess(accessType, task);
				registered = true;
			}
		}
		
		if (upgraded) {
			bool erased = false;
			address_list_t::iterator it;
			for (it = reads.begin(); !erased && it != reads.end(); ++it) {
				if (*it == address) {
					reads.erase(it);
					writes.push_back(address);
					erased = true;
				}
			}
			assert(erased);
		}
		
		if (registered) {
			if (accessType == READ_ACCESS_TYPE) {
				reads.push_back(address);
			} else {
				writes.push_back(address);
			}
		}
		
		if (becameUnsatisfied) {
			task->increasePredecessors();
		}
	}
	
	void finalizeDataAccess(Task *task, DataAccessType accessType, void *address, CPUDependencyData &hpDependencyData) {
		Task *parentTask = task->getParent();
		assert(parentTask != nullptr);
		
		TaskDataAccesses &parentAccessStruct = parentTask->getDataAccesses();
		assert(!parentAccessStruct.hasBeenDeleted());
		addresses_t &addresses = parentAccessStruct._dataAccessSequences;
		
		DataAccessSequence *sequence = nullptr;
		{
			std::lock_guard<TaskDataAccesses::spinlock_t> guard(parentAccessStruct._lock);
			
			const addresses_t::const_iterator it = addresses.find(address);
			assert(it != addresses.end());
			
			sequence = it->second;
			assert(sequence != nullptr);
		}
		
		std::lock_guard<DataAccessSequence::spinlock_t> guard(sequence->_lock);
		sequence->finalizeDataAccess(task, accessType, hpDependencyData._satisfiedOriginators);
	}
	
	bool registerTaskDataAccesses(Task *task, __attribute__((unused)) ComputePlace *computePlace)
	{
		assert(task != nullptr);
		
		// Enable the wait clause to release the dependencies once all children finish
		task->setDelayedRelease(true);
		
		nanos6_task_info_t *taskInfo = task->getTaskInfo();
		assert(taskInfo != 0);
		
		task->increasePredecessors(2);
		
		// This part creates the DataAccesses and inserts it to dependency system
		taskInfo->register_depinfo(task->getArgsBlock(), task);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		if (accessStructures.hasDataAccesses()) {
			task->increaseRemovalBlockingCount();
		}
		
		return task->decreasePredecessors(2);
	}
	
	void unregisterTaskDataAccesses(Task *task, ComputePlace *computePlace)
	{
		assert(task != nullptr);
		
		TaskDataAccesses &accessStruct = task->getDataAccesses();
		assert(!accessStruct.hasBeenDeleted());
		
		if (!accessStruct.hasDataAccesses()) return;
		
		CPUDependencyData localDependencyData;
		CPUDependencyData &hpDependencyData = (computePlace != nullptr) ?
				computePlace->getDependencyData() : localDependencyData;
		
#ifndef NDEBUG
		{
			bool alreadyTaken = false;
			assert(hpDependencyData._inUse.compare_exchange_strong(alreadyTaken, true));
		}
#endif
		
		{
			std::lock_guard<TaskDataAccesses::spinlock_t> guard(accessStruct._lock);
			
			for (void *address : accessStruct._readAccessAddresses) {
				finalizeDataAccess(task, READ_ACCESS_TYPE, address, hpDependencyData);
			}
			
			for (void *address : accessStruct._writeAccessAddresses) {
				finalizeDataAccess(task, WRITE_ACCESS_TYPE, address, hpDependencyData);
			}
			
			task->decreaseRemovalBlockingCount();
		}
		
		processSatisfiedOriginators(hpDependencyData, computePlace, false);
		
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
	
	void handleEnterTaskwait(Task *task, __attribute__((unused)) ComputePlace *computePlace)
	{
		assert(task != nullptr);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		if (accessStructures.hasDataAccesses()) {
			task->decreaseRemovalBlockingCount();
		}
	}
	
	
	void handleExitTaskwait(Task *task, __attribute__((unused)) ComputePlace *computePlace)
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
	}
};

#pragma GCC visibility pop

