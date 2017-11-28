/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include <nanos6.h>
#include <tasks/Task.hpp>
#include <executors/threads/WorkerThread.hpp>

#include "DataAccess.hpp"
#include "TaskDataAccessesImplementation.hpp"
#include "ReductionInfo.hpp"

void *nanos_get_reduction_storage(void *original) {
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);
	
	Task *task = currentThread->getTask();
	assert(task != nullptr);
	
	DataAccess *dataAccess = nullptr;
	TaskDataAccesses::accesses_t &accesses = task->getDataAccesses()._accesses;
	accesses.processIntersecting(
		DataAccessRegion(original, /* length */ 1),
		[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
			assert(dataAccess == nullptr); // This intersection should only match once
			
			dataAccess = &(*position);
			assert(dataAccess != nullptr);
			
			return true;
		}
	);
	
	assert(dataAccess != nullptr);
	
	CPU *currentCPU = currentThread->getComputePlace();
	size_t cpuId = currentCPU->_virtualCPUId;
	
	ReductionInfo *reductionInfo = dataAccess->getReductionInfo();
	assert(reductionInfo != nullptr);
	
	void *address = reductionInfo->getCPUPrivateStorage(cpuId).getStartAddress();
	
	return address;
}
