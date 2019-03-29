/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef SCHEDULER_INTERFACE_HPP
#define SCHEDULER_INTERFACE_HPP


#include <atomic>
#include <string>
#include "hardware/places/ComputePlace.hpp"

class Task;


//! \brief Interface that schedulers must implement
class SchedulerInterface {
public:
	//! \brief An object to allow the scheduler to push tasks directly to a thread
	struct polling_slot_t {
		std::atomic<Task *> _task;
		
		//! \brief scheduler-dependent information
		union {
			void *_schedulerInfo;
			std::atomic<void *> _atomicSchedulerInfo;
		};
		
		polling_slot_t()
			: _task(nullptr), _schedulerInfo(nullptr)
		{
		}
		
		Task *getTask()
		{
			Task *result = _task.load();
			while(!_task.compare_exchange_strong(result, nullptr)) {}
			return result;
		}
		
		bool setTask(Task *task)
		{
			Task *expected = nullptr;
			bool success = _task.compare_exchange_strong(expected, task);
			return success;
		}
	};
	
	
	enum ReadyTaskHint {
		NO_HINT,
		CHILD_TASK_HINT,
		SIBLING_TASK_HINT,
		BUSY_COMPUTE_PLACE_TASK_HINT,
		UNBLOCKED_TASK_HINT
	};

	
	
	virtual ~SchedulerInterface()
	{
	}
	
	//! \brief Add a (ready) task that has been created or freed (but not unblocked)
	//!
	//! \param[in] task the task to be added
	//! \param[in] computePlace the hardware place of the creator or the liberator
	//! \param[in] hint a hint about the relation of the task to the current task
	//!
	//! \returns an idle ComputePlace that is to be resumed or nullptr
	virtual ComputePlace *addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint, bool doGetIdle = true) = 0;
	
	//! \brief Get a ready task for execution
	//!
	//! \param[in] computePlace the hardware place asking for scheduling orders
	//! \param[in] currentTask a task within whose context the resulting task will run
	//! \param[in] canMarkAsIdle true if the scheduler should mark the computePlace as idle if there are no pending tasks
	//! \param[in] doWait true if the scheduler should poll from some time before returning
	//!
	//! \returns a ready task or nullptr
	virtual Task *getReadyTask(ComputePlace *computePlace, Task *currentTask = nullptr, bool canMarkAsIdle = true, bool doWait = false) = 0;
	
	//! \brief Check if it can wait, or if it has to wait using polling slots
	//!
	//! \returns True if it can wait.
	virtual bool canWait();
	
	//! \brief Get an idle hardware place
	//!
	//! \param[in] force indicates that an idle hardware place must be returned (if any) even if the scheduler does not have any pending work to be assigned
	//!
	//! \returns a hardware place that becomes non idle or nullptr
	virtual ComputePlace *getIdleComputePlace(bool force=false) = 0;
	
	//! \brief Notify the scheduler that a hardware place is being disabled so that it has a chance to migrate any preassigned tasks
	//! 
	//! \param[in] computePlace the hardware place that is about to be disabled
	//! 
	//! This method has a default implementation that does nothing
	virtual void disableComputePlace(ComputePlace *computePlace);
	
	//! \brief Notify the scheduler that a hardware place is back online so that it preassign tasks to it
	//! 
	//! \param[in] computePlace the hardware place that is about to be enabled
	//! 
	//! This method has a default implementation that does nothing
	virtual void enableComputePlace(ComputePlace *computePlace);
	
	//! \brief Attempt to get a one task polling slot
	//! 
	//! \param[in] computePlace the hardware place asking for scheduling orders
	//! \param[out] pollingSlot a pointer to a location that the caller will poll for ready tasks
	//! 
	//! \returns true if the caller is allowed to poll that memory position for a single ready task or if it actually got a task, otherwise false and the hardware place is assumed to become idle
	//! 
	//! This method has a default implementation that just falls back to getReadyTask.
	virtual bool requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, bool canMarkAsIdle = true);
	
	//! \brief Attempt to release the polling slot
	//! 
	//! \param[in] computePlace the hardware place asking for scheduling orders
	//! \param[out] pollingSlot a pointer to a location that the caller is polling for ready tasks
	//! 
	//! \returns true if the caller has successfully released the polling slot otherwise false indicating that there already is a taskl assigned or it is on the way
	//! 
	//! This method has a default implementation that matches the default implementation of requestPolling.
	virtual bool releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, bool canMarkAsIdle = true);
	
	virtual std::string getName() const = 0;
};


#endif // SCHEDULER_INTERFACE_HPP
