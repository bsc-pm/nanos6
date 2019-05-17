/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "executors/threads/CPUActivation.hpp"
#include "system/PollingAPI.hpp"
#include "system/RuntimeInfo.hpp"
#include "tasks/Task.hpp"

#include "SchedulerGenerator.hpp"
#include "Scheduler.hpp"
#include "SchedulerInterface.hpp"


SchedulerInterface *Scheduler::_scheduler;


void Scheduler::initialize()
{
	_scheduler = SchedulerGenerator::createHostScheduler();
	RuntimeInfo::addEntry("scheduler", "Scheduler", _scheduler->getName());
}

void Scheduler::shutdown() 
{
	delete _scheduler;
}

Task *Scheduler::getReadyTask(ComputePlace *computePlace, Task *currentTask, bool canMarkAsIdle, bool doWait)
{
	assert(computePlace != nullptr);
	Task *task = nullptr;
	
	if (_scheduler->canWait() || !doWait) {
		task = _scheduler->getReadyTask(computePlace, currentTask, canMarkAsIdle, doWait);
	} else {
		polling_slot_t pollingSlot;
		
		if (_scheduler->requestPolling(computePlace, &pollingSlot, canMarkAsIdle)) {
			Instrument::threadEnterBusyWait(Instrument::scheduling_polling_slot_busy_wait_reason);
			while ((task == nullptr) && !ThreadManager::mustExit() && CPUActivation::acceptsWork((CPU *)computePlace)) {
				// Keep trying
				task = pollingSlot.getTask();
				if (task == nullptr) {
					PollingAPI::handleServices();
				}
			}
			Instrument::threadExitBusyWait();
			
			if (ThreadManager::mustExit()) {
				__attribute__((unused)) bool worked = _scheduler->releasePolling(computePlace, &pollingSlot, canMarkAsIdle);
				assert(worked);
			}
			
			if (!CPUActivation::acceptsWork((CPU *)computePlace)) {
				// The CPU is about to be disabled
				
				// Release the polling slot
				_scheduler->releasePolling(computePlace, &pollingSlot, canMarkAsIdle);
				
				// We may already have a task assigned through
				task = pollingSlot.getTask();
			}
		}
	}
	
	return task;
}

