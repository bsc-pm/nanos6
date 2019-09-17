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
#include "TaskDataAccesses.hpp"
#include "ReductionInfo.hpp"

void *nanos6_get_reduction_storage1(void *original,
		__attribute__((unused)) long dim1size,
		__attribute__((unused)) long dim1start,
		__attribute__((unused)) long dim1end)
{
	assert(dim1start == 0L);

	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);

	Task *task = currentThread->getTask();
	assert(task != nullptr);
	Task *child = task;
	Task *parent = task->getParent();
	TaskDataAccesses &childAccessStruct = child->getDataAccesses();
	TaskDataAccesses &parentAccessStruct = parent != nullptr ? parent->getDataAccesses() : child->getDataAccesses();

	FatalErrorHandler::failIf(task->getDeviceType() != nanos6_device_t::nanos6_host_device,
			"Device reductions are not supported");

	ReductionInfo *reductionInfo = nullptr;
	long int slotIndex = -1;

	CPU *currentCPU = currentThread->getComputePlace();
	size_t cpuId = currentCPU->getIndex();

	assert(!task->isTaskfor() || (task->getParent() != nullptr));
	bool getParentAccesses = task->isTaskfor() && task->getParent()->isTaskfor();

	// Taskloop must be final.
	assert(!(task->isTaskfor() && !parent->isFinal()));

	TaskDataAccesses &accessStruct =
		getParentAccesses ? parentAccessStruct : childAccessStruct;

	task = getParentAccesses ? task->getParent() : task;

	DataAccess * access = accessStruct.findAccess(original);

	bool weak = access->isWeak();

	if(reductionInfo == nullptr && !weak) {
		assert(slotIndex == -1);
		reductionInfo = access->getReductionInfo();
		slotIndex = reductionInfo->getFreeSlotIndex(cpuId);
		assert(slotIndex >= 0);
	}

	assert(reductionInfo != nullptr || weak);
	assert(slotIndex >= 0 || weak);

	void *address = original;
	if (reductionInfo != nullptr) {
		const void * originalAddress = reductionInfo->getOriginalAddress();
		__attribute__((unused)) const size_t originalLength = reductionInfo->getOriginalLength();
		assert(((char*)original) >= ((char*)originalAddress));
		assert(((char*)original) < (((char*)originalAddress)
					+ originalLength));

		address = ((char*)reductionInfo->getFreeSlotStorage(slotIndex)) +
			((char*)original - (char*)originalAddress);
	}

	return address;
}

