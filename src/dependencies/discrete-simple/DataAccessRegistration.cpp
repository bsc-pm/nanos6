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
	typedef TaskDataAccesses::addresses_map_t addresses_map_t;
	typedef TaskDataAccesses::addresses_vec_t addresses_vec_t;
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
			
			Scheduler::addReadyTask(
				satisfiedOriginator,
				computePlaceHint,
				(fromBusyThread ?
					SchedulerInterface::SchedulerInterface::BUSY_COMPUTE_PLACE_TASK_HINT
					: SchedulerInterface::SchedulerInterface::SIBLING_TASK_HINT
				)
			);
		}
		
		hpDependencyData._satisfiedOriginators.clear();
	}
	
	void registerTaskDataAccess(Task *task, DataAccessType accessType, void *address) {
		assert(task != nullptr);
		assert(address != nullptr);
		
        TaskDataAccesses &accessStruct = task->getDataAccesses();
        address_list_t * addresses = accessStruct._accessAddresses;

        // Couple access type with the address. The last bit is 0 if READ, 1 if WRITE.
        void * address_typed = (void *)((uintptr_t) address | (uintptr_t)(accessType != READ_ACCESS_TYPE));

        address_list_t::iterator it;
        for(it = addresses->begin(); it != addresses->end(); it++) {
            // Deactivate last bit of *it just in case the previous access was write and current is read, so the condition is not satisfied falsely.
            void * read_it = (void *)((uintptr_t) *it & (uintptr_t)~1);
            if(read_it > address)
                break;
            if(*it == address || *it == address_typed) {
                if(*it < address_typed) 
                    *it = address_typed;
                return;
            }
        }
        for(address_list_t::iterator aux = addresses->begin(); aux != addresses->end(); aux++) {
            assert(*aux != address_typed);
        }
        addresses->insert(it, address_typed);
	}
	
	void finalizeDataAccess(Task *task, DataAccessType accessType, void *address, CPUDependencyData &hpDependencyData) {
        TaskDataAccesses &accessStruct = task->getDataAccesses();
        DataAccessSequence *sequence = nullptr;
        if(accessStruct._map) {
            addresses_map_t * addresses = accessStruct._dataAccessSequencesMap;
            sequence = addresses->at(address);
        }
        else {
            addresses_vec_t * addresses = accessStruct._dataAccessSequencesVec;
            for(addresses_vec_t::iterator it = addresses->begin(); it != addresses->end(); it++) {
                if(it->first == address)
                    sequence = it->second;
            }
        }
        assert(sequence != nullptr);
		
		std::lock_guard<DataAccessSequence::spinlock_t> guard(sequence->_lock);
		sequence->finalizeDataAccess(task, accessType, hpDependencyData._satisfiedOriginators);
	}
	
	bool registerTaskDataAccesses(Task *task, __attribute__((unused)) ComputePlace *computePlace, __attribute__((unused)) CPUDependencyData &hpDependencyData)
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

        insertAccesses(task);

		assert(!accessStructures.hasBeenDeleted());
		if (accessStructures.hasDataAccesses()) {
			task->increaseRemovalBlockingCount();
		}
		
		return task->decreasePredecessors(2);
	}
	
	void unregisterTaskDataAccesses(Task *task, ComputePlace *computePlace, __attribute__((unused)) CPUDependencyData &hpDependencyData)
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
			for (void *address_typed : *accessStruct._accessAddresses) {
                void * address = (void *)((uintptr_t) address_typed & (uintptr_t)~(1));
                bool write = ((uintptr_t)address_typed & 1);
                DataAccessType accessType = !write ? READ_ACCESS_TYPE : WRITE_ACCESS_TYPE;
				finalizeDataAccess(task, accessType, address, hpDependencyData);
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
	
	void handleEnterTaskwait(Task *task, __attribute__((unused)) ComputePlace *computePlace, __attribute__((unused)) CPUDependencyData &dependencyData)
	{
		assert(task != nullptr);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		if (accessStructures.hasDataAccesses()) {
			task->decreaseRemovalBlockingCount();
		}
	}
	
	
	void handleExitTaskwait(Task *task, __attribute__((unused)) ComputePlace *computePlace, __attribute__((unused)) CPUDependencyData &dependencyData)
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

    void insertAccesses(Task * task)
    {
		TaskDataAccesses &accessStruct = task->getDataAccesses();
		assert(!accessStruct.hasBeenDeleted());
		
		Task *parentTask = task->getParent();
		assert(parentTask != nullptr);
		
		TaskDataAccesses &parentAccessStruct = parentTask->getDataAccesses();
		assert(!parentAccessStruct.hasBeenDeleted());

        // Get all seqs
        std::vector<std::pair<void *, DataAccessSequence *> > seqs(accessStruct._accessAddresses->size());
        size_t seq_index = 0;
        for (void *address_typed : *accessStruct._accessAddresses) {
            void * address = (void *)((uintptr_t) address_typed & (uintptr_t)~(1));
            DataAccessSequence *sequence = nullptr;

            if(!parentAccessStruct._map) { 
                addresses_vec_t * addresses = parentAccessStruct._dataAccessSequencesVec;
                for(addresses_vec_t::iterator it = addresses->begin(); it != addresses->end(); it++) {
                    if(it->first == address)
                        sequence = it->second;
                }
                //! If there is free space in the vector, add it. Otherwise, convert vector in map.
                if(sequence == nullptr) {
                    sequence = MemoryAllocator::newObject<DataAccessSequence>();
                    sequence->incrementRemaining();
                    if(addresses->size() < addresses->capacity()) {
                        addresses->push_back({address, sequence});
                    }
                    else {
                        parentAccessStruct.vecToMap();
                        assert(parentAccessStruct._map && parentAccessStruct._dataAccessSequencesMap != nullptr);
                        (*parentAccessStruct._dataAccessSequencesMap)[address] = sequence;
                    }
                    assert(sequence != nullptr);
                }
            }
            else {
                addresses_map_t * addresses = parentAccessStruct._dataAccessSequencesMap;
                addresses_map_t::iterator it = addresses->find(address);
                if (it != addresses->end()) {
                    sequence = it->second;
                } else {
                    sequence = MemoryAllocator::newObject<DataAccessSequence>();
                    (*addresses)[address] = sequence;
                }
            }
            assert(sequence != nullptr);
            seqs[seq_index++] = {address, sequence};
        }
        if(accessStruct._map) {
            accessStruct._dataAccessSequencesMap->reserve(seqs.size());
            accessStruct._dataAccessSequencesMap->insert(seqs.begin(), seqs.end());
        }
        else
            accessStruct._dataAccessSequencesVec->insert(accessStruct._dataAccessSequencesVec->end(), seqs.begin(), seqs.end());
        assert(seq_index == accessStruct._accessAddresses->size() && seq_index == seqs.size());

        // Process all seqs
        seq_index = 0;
        for (void *address_typed : *accessStruct._accessAddresses) {
            bool write = ((uintptr_t)address_typed & 1);
            DataAccessType accessType = !write ? READ_ACCESS_TYPE : WRITE_ACCESS_TYPE;
            DataAccessSequence *sequence = seqs[seq_index++].second;
            bool becameUnsatisfied = false;
            {
                std::lock_guard<DataAccessSequence::spinlock_t> guardSeq(sequence->_lock);

                if (sequence->registeredLastDataAccess(task)) {
                    if (accessType != READ_ACCESS_TYPE) {
                        DataAccessType prevAccessType;
                        becameUnsatisfied = sequence->upgradeLastDataAccess(&prevAccessType);
                    }
                } else {
                    becameUnsatisfied = !sequence->registerDataAccess(accessType, task);
                }
            }

            if (becameUnsatisfied) {
                task->increasePredecessors();
            }
        }
    }
};

#pragma GCC visibility pop

