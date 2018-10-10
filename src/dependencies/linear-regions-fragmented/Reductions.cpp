/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include <nanos6.h>
#include <tasks/Task.hpp>
#include <executors/threads/WorkerThread.hpp>

#include <InstrumentReductions.hpp>

#include "DataAccess.hpp"
#include "TaskDataAccessesImplementation.hpp"
#include "ReductionInfo.hpp"

void *nanos6_get_reduction_storage1(void *original,
		long dim1size,
		__attribute__((unused)) long dim1start,
		__attribute__((unused)) long dim1end)
{
	assert(dim1start == 0L);
	
	Instrument::enterRetrievePrivateReductionStorage(
		DataAccessRegion(original, dim1size)
	);
	
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);
	
	Task *task = currentThread->getTask();
	assert(task != nullptr);
	
	FatalErrorHandler::failIf(task->getDeviceType() != nanos6_device_t::nanos6_host_device,
			"Device reductions are not supported");
	
	ReductionInfo *reductionInfo = nullptr;
	long int slotIndex = -1;
	
	CPU *currentCPU = currentThread->getComputePlace();
	size_t cpuId = currentCPU->_virtualCPUId;
	
	TaskDataAccesses &taskAccesses =
		task->isTaskloop() ? task->getParent()->getDataAccesses() : task->getDataAccesses();
	
	// Need the lock, as access can be fragmented while we access it
	taskAccesses._lock.lock();
	
	TaskDataAccesses::accesses_t &accesses = taskAccesses._accesses;
	accesses.processIntersecting(
		DataAccessRegion(original, dim1size),
		[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
			DataAccess *dataAccess = &(*position);
			
			if (dataAccess->getType() != REDUCTION_ACCESS_TYPE) {
				assert(task->isFinal());
				assert((dataAccess->getType() == READWRITE_ACCESS_TYPE) ||
						(dataAccess->getType() == WRITE_ACCESS_TYPE) ||
						(dataAccess->getType() == CONCURRENT_ACCESS_TYPE));
				assert(reductionInfo == nullptr);
				
				return false;
			}
			
			// If reduction is registered, obtain the corresponding reduction
			// storage and reduction free slot
			if (reductionInfo == nullptr) {
				assert(slotIndex == -1);
				reductionInfo = dataAccess->getReductionInfo();
				slotIndex = reductionInfo->getFreeSlotIndex(cpuId);
			}
			
			assert(dataAccess->getReductionInfo() == reductionInfo);
			assert(slotIndex >= 0);
			
			// Register assigned slot in the data access
			dataAccess->setReductionAccessedSlot(slotIndex);
			
			return true;
		}
	);
	
	taskAccesses._lock.unlock();
	
	void *address = original;
	if (reductionInfo != nullptr) {
		const DataAccessRegion& originalFullRegion = reductionInfo->getOriginalRegion();
		assert(((char*)original) >= ((char*)originalFullRegion.getStartAddress()));
		assert(((char*)original) < (((char*)originalFullRegion.getStartAddress())
					+ originalFullRegion.getSize()));
		
		address = ((char*)reductionInfo->getFreeSlotStorage(slotIndex).getStartAddress()) +
			((char*)original - (char*)originalFullRegion.getStartAddress());
	}
	
	Instrument::exitRetrievePrivateReductionStorage(
		*reductionInfo,
		DataAccessRegion(address, dim1size),
		DataAccessRegion(original, dim1size)
	);
	
	return address;
}
