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


template <DataAccessType ACCESS_TYPE, bool WEAK, typename... ReductionInfo>
void register_access(void *handler, void *start, size_t length, ReductionInfo... reductionInfo)
{
	assert(handler != 0);
	Task *task = (Task *) handler;
	
	if (WEAK && task->isTaskloop()) {
		std::cerr << "Warning: task loop cannot have weak dependencies. Changing them to strong dependencies." << std::endl;
	}
	
	Instrument::registerTaskAccess(task->getInstrumentationTaskId(), ACCESS_TYPE, WEAK && !task->isFinal() && !task->isTaskloop(), start, length);
	
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != 0); // NOTE: The "main" task is not created by a WorkerThread, but in any case it is not supposed to have dependencies
	
	DataAccessRegion accessRegion(start, length);
	
	DataAccessSequence *accessSequence = 0;
	Task *parent = task->getParent();
	if (parent != 0) {
		for (DataAccessBase &parentAccessBase : parent->getDataAccesses()) {
			DataAccess &parentAccess = (DataAccess &) parentAccessBase;
			DataAccessSequence *parentSequence = parentAccess._dataAccessSequence;
			assert(parentSequence != 0);
			
			if (parentSequence->_accessRegion == accessRegion) {
				accessSequence = &parentAccess._subaccesses;
				break;
			}
		}
	}
	
	if (accessSequence == 0) {
		// An access that is not a subset of the parent accesses, therefore
		// (if the code is correct) it must be temporary data created by the parent
		DependencyDomain *dependencyDomain = currentWorkerThread->getDependencyDomain();
		accessSequence = & (*dependencyDomain)[accessRegion]._accessSequence;
	}
	
	DataAccess *dataAccess;
	bool canStart = DataAccessRegistration::registerTaskDataAccess(task, ACCESS_TYPE, WEAK && !task->isFinal() && !task->isTaskloop(),
			accessSequence, /* OUT */ dataAccess, reductionInfo...);
	
	if (dataAccess != 0) {
		// A new data access, as opposed to a repeated or upgraded one
		task->getDataAccesses().push_back(*dataAccess);
	}
	
	if (!canStart) {
		task->increasePredecessors();
	}
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


void nanos6_register_concurrent_depinfo(void *handler, void *start, size_t length)
{
	register_access<CONCURRENT_ACCESS_TYPE, false>(handler, start, length);
}

void nanos6_register_region_reduction_depinfo1(
		int reduction_operation, int reduction_index,
		void *handler,
		int symbol_index,
		char const *region_text,
		void *base_address,
		long dim1size, long dim1start, long dim1end)
{
	FatalErrorHandler::failIf(dim1size > sizeof(long long), "Array reductions not supported");

	register_access<REDUCTION_ACCESS_TYPE, false>(handler, base_address, dim1size, reduction_operation);
}
