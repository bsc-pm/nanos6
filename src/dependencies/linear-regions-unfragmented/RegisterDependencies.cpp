/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include <InstrumentDependenciesByAccess.hpp>

#include <nanos6.h>
#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include "../DataAccessType.hpp"
#include "DataAccessRegistration.hpp"
#include "LinearRegionDataAccessMap.hpp"


template <DataAccessType ACCESS_TYPE, bool WEAK>
void register_access(void *handler, void *start, size_t length)
{
	assert(handler != 0);
	Task *task = (Task *) handler;
	
	Instrument::registerTaskAccess(task->getInstrumentationTaskId(), ACCESS_TYPE, WEAK && !task->isFinal(), start, length);
	
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != 0); // NOTE: The "main" task is not created by a WorkerThread, but in any case it is not supposed to have dependencies
	
	DataAccessRegion accessRegion(start, length);
	
	LinearRegionDataAccessMap *topMap = nullptr;
	LinearRegionDataAccessMap *bottomMap = nullptr;
	DataAccess *superAccess = nullptr;
	SpinLock *lock = nullptr;
	
	Task *parent = task->getParent();
	if (parent != 0) {
		for (DataAccessBase &parentAccessBase : parent->getDataAccesses()) {
			DataAccess &parentAccess = (DataAccess &) parentAccessBase;
			DataAccessRegion intersection = parentAccess._region.intersect(accessRegion);
			
			if (!intersection.empty()) {
				assert("A subaccess cannot extend beyond that of its parent access" && (intersection == accessRegion));
				topMap = &parentAccess._topSubaccesses;
				bottomMap = &parentAccess._bottomSubaccesses;
				
				superAccess = &parentAccess;
				assert(superAccess != nullptr);
				
				lock = parentAccess._lock;
				assert(lock != nullptr);
				
				break;
			}
		}
	}
	
	if (bottomMap == 0) {
		// An access that is not a subset of the parent accesses, therefore
		// (if the code is correct) it must be temporary data created by the parent
		DependencyDomain *domain = currentWorkerThread->getDependencyDomain();
		assert(domain != nullptr);
		
		bottomMap = &domain->_map;
		lock = &domain->_lock;
	}
	
	DataAccessRegistration::registerTaskDataAccess(task, ACCESS_TYPE, WEAK && !task->isFinal(), accessRegion, superAccess, lock, topMap, bottomMap);
}


void nanos6_register_read_depinfo(void *handler, void *start, size_t length)
{
	register_access<READ_ACCESS_TYPE, false>(handler, start, length);
}


void nanos6_register_write_depinfo(void *handler, void *start, size_t length)
{
	register_access<WRITE_ACCESS_TYPE, false>(handler, start, length);
}


void nanos6_register_readwrite_depinfo(void *handler, void *start, size_t length)
{
	register_access<READWRITE_ACCESS_TYPE, false>(handler, start, length);
}


void nanos6_register_weak_read_depinfo(void *handler, void *start, size_t length)
{
	register_access<READ_ACCESS_TYPE, true>(handler, start, length);
}


void nanos6_register_weak_write_depinfo(void *handler, void *start, size_t length)
{
	register_access<WRITE_ACCESS_TYPE, true>(handler, start, length);
}


void nanos6_register_weak_readwrite_depinfo(void *handler, void *start, size_t length)
{
	register_access<READWRITE_ACCESS_TYPE, true>(handler, start, length);
}


