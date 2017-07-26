/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "SchedulerInterface.hpp"

#include "Scheduler.hpp"

#include <cassert>

void SchedulerInterface::disableComputePlace(__attribute__((unused)) ComputePlace *computePlace)
{
}


void SchedulerInterface::enableComputePlace(__attribute__((unused)) ComputePlace *computePlace)
{
}


bool SchedulerInterface::requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot)
{
	assert(pollingSlot != nullptr);
	assert(pollingSlot->_task == nullptr);
	
	// Default implementation: attempt to get a ready task and fail if not possible
	Task *task = Scheduler::getReadyTask(computePlace);
	
	if (task != nullptr) {
		Task *expected = nullptr;
		
		pollingSlot->_task.compare_exchange_strong(expected, task);
		assert(expected == nullptr);
		
		return true;
	} else {
		return false;
	}
}


bool SchedulerInterface::releasePolling(__attribute__((unused)) ComputePlace *computePlace, __attribute__((unused)) polling_slot_t *pollingSlot)
{
	// The default implementation should never be called if there is a default implementation of requestPolling
	// otherwise there should be an implementation of this method that matches requestPolling
	return (pollingSlot->_task.load(std::memory_order_seq_cst) == nullptr);
}


#include "instrument/support/InstrumentThreadLocalDataSupport.hpp"
#include "instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp"
