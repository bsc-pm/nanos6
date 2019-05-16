/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef SCHEDULER_HPP
#define SCHEDULER_HPP

#include <atomic>
#include <cassert>

#include "SchedulerInterface.hpp"
#include "executors/threads/CPUManager.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include <HardwareCounters.hpp>
#include <InstrumentTaskStatus.hpp>
#include <Monitoring.hpp>


class HardwareDescription;
class ComputePlace;


//! \brief This class is the main interface within the runtime to interact with the scheduler
//!
//! It holds a pointer to the actual scheduler and forwards the calls to it.
class Scheduler {
	static SchedulerInterface *_scheduler;
	
public:
	//! \brief An object to allow the scheduler to push tasks directly to a thread
	typedef SchedulerInterface::polling_slot_t polling_slot_t;
	
	//! \brief Initializes the _scheduler member and in turn calls its initialization method
	static void initialize();

	static void shutdown();
	
	//! \brief Add a (ready) task that has been created or freed (but not unblocked)
	//!
	//! \param[in] task the task to be added
	//! \param[in] computePlace the hardware place of the creator or the liberator
	//! \param[in] hint a hint about the relation of the task to the current task
	//!
	//! \returns an idle ComputePlace that is to be resumed or nullptr
	static inline ComputePlace *addReadyTask(Task *task, ComputePlace *computePlace, SchedulerInterface::ReadyTaskHint hint = SchedulerInterface::NO_HINT)
	{
		assert(task != 0);
		Instrument::taskIsReady(task->getInstrumentationTaskId());
		
		ComputePlace *cpu = nullptr;
		if (task->getThread() != nullptr) {
			cpu = task->getThread()->getComputePlace();
		}
		
		HardwareCounters::stopTaskMonitoring(task);
		Monitoring::taskChangedStatus(task, ready_status, cpu);
		
		if (hint == SchedulerInterface::UNBLOCKED_TASK_HINT) {
			return _scheduler->addReadyTask(task, computePlace, hint, false);
		} else if (task->isTaskloop()) {
			_scheduler->addReadyTask(task, computePlace, hint, false);
			
			std::vector<CPU *> idleCPUs;
			CPUManager::getIdleCPUs(idleCPUs);
			if (!idleCPUs.empty()) {
				ThreadManager::resumeIdle(idleCPUs);
			}
			
			return nullptr;
		} else {
			return _scheduler->addReadyTask(task, computePlace, hint);
		}
	}
	
	//! \brief Get a ready task for execution
	//!
	//! \param[in] computePlace the hardware place asking for scheduling orders
	//! \param[in] currentTask a task within whose context the resulting task will run
	//! \param[in] canMarkAsIdle true if the scheduler should mark the computePlace as idle if there are no pending tasks
	//! \param[in] doWait true if the scheduler should poll from some time before returning
	//!
	//! \returns a ready task or nullptr
	static Task *getReadyTask(ComputePlace *computePlace, Task *currentTask = nullptr, bool canMarkAsIdle = true, bool doWait = false);
	
	//! \brief Get an idle hardware place
	//!
	//! \param[in] force idicates that an idle hardware place must be returned (if any) even if the scheduler does not have any pending work to be assigned
	//!
	//! \returns a hardware place that becomes non idle or nullptr
	static inline ComputePlace *getIdleComputePlace(bool force=false)
	{
		return _scheduler->getIdleComputePlace(force);
	}
	
	//! \brief Notify the scheduler that a hardware place is being disabled so that it has a chance to migrate any preassigned tasks
	//! 
	//! \param[in] computePlace the hardware place that is about to be disabled
	static void disableComputePlace(ComputePlace *computePlace)
	{
		_scheduler->disableComputePlace(computePlace);
	}
	
	//! \brief Notify the scheduler that a hardware place is back online so that it preassign tasks to it
	//! 
	//! \param[in] computePlace the hardware place that is about to be enabled
	static void enableComputePlace(ComputePlace *computePlace)
	{
		_scheduler->enableComputePlace(computePlace);
	}
};


#endif // SCHEDULER_HPP
