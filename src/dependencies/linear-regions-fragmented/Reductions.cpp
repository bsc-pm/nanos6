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

void *nanos6_get_reduction_storage1(void *original,
		long dim1size,
		__attribute__((unused)) long dim1start,
		__attribute__((unused)) long dim1end)
{
	assert(dim1start == 0L);
	
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);
	
	Task *task = currentThread->getTask();
	assert(task != nullptr);
	
	DataAccess *firstAccess = nullptr;
	
	CPU *currentCPU = currentThread->getComputePlace();
	size_t cpuId = currentCPU->_virtualCPUId;
	
	TaskDataAccesses &taskAccesses =
		task->isTaskloop() ? task->getParent()->getDataAccesses() : task->getDataAccesses();
	
	// Need the lock, as access can be fragmented while we access it
	std::lock_guard<TaskDataAccesses::spinlock_t> guard(taskAccesses._lock);
	
	TaskDataAccesses::accesses_t &accesses = taskAccesses._accesses;
	accesses.processIntersecting(
		DataAccessRegion(original, dim1size),
		[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
			DataAccess *dataAccess = &(*position);
			
			if (dataAccess->getType() != REDUCTION_ACCESS_TYPE)
			{
				assert(task->isFinal());
				assert((dataAccess->getType() == READWRITE_ACCESS_TYPE) ||
						(dataAccess->getType() == WRITE_ACCESS_TYPE) ||
						(dataAccess->getType() == CONCURRENT_ACCESS_TYPE));
				assert(firstAccess == nullptr);
				
				return false;
			}
			
			assert(dataAccess->getType() == REDUCTION_ACCESS_TYPE);
			
			dataAccess->setReductionCpu(cpuId);
			
			if ((firstAccess == nullptr) ||
					(firstAccess->getAccessRegion().getStartAddress() <
					dataAccess->getAccessRegion().getStartAddress()))
				firstAccess = dataAccess;
			
			return true;
		}
	);
	
	void *address = original;
	
	// If reduction is registered, obtain the corresponding reduction storage
	if (firstAccess != nullptr)
	{
		ReductionInfo *reductionInfo = firstAccess->getReductionInfo();
		assert(reductionInfo != nullptr);

		assert(((char*)original) >= ((char*)reductionInfo->getOriginalRegion().getStartAddress()));
		assert(((char*)original) < (((char*)reductionInfo->getOriginalRegion().getStartAddress())
					+ reductionInfo->getOriginalRegion().getSize()));

		address = ((char*)reductionInfo->getCPUPrivateStorage(cpuId).getStartAddress()) +
			((char*)original - (char*)reductionInfo->getOriginalRegion().getStartAddress());
	}
	
	return address;
}
