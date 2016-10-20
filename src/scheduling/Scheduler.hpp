#ifndef SCHEDULER_HPP
#define SCHEDULER_HPP


#include "SchedulerInterface.hpp"

#include <InstrumentTaskStatus.hpp>
#include <tasks/Task.hpp>

#include <atomic>
#include <cassert>


class HardwareDescription;
class ComputePlace;

class Task;


//! \brief This class is the main interface within the runtime to interact with the scheduler
//!
//! It holds a pointer to the actual scheduler and forwards the calls to it.
class Scheduler {
	static SchedulerInterface *_scheduler;
	
public:
	//! \brief Initializes the _scheduler member and in turn calls its initialization method
	static void initialize();
	
	//! \brief This is needed to make the scheduler aware of the CPUs that are online
	static inline SchedulerInterface *getInstance()
	{
		assert(_scheduler != nullptr);
		return _scheduler;
	}
	
	//! \brief Add a (ready) task that has been created or freed (but not unblocked)
	//!
	//! \param[in] task the task to be added
	//! \param[in] hardwarePlace the hardware place of the creator or the liberator
	//! \param[in] hint a hint about the relation of the task to the current task
	//!
	//! \returns an idle ComputePlace that is to be resumed or nullptr
	static inline ComputePlace *addReadyTask(Task *task, ComputePlace *hardwarePlace, SchedulerInterface::ReadyTaskHint hint = SchedulerInterface::NO_HINT)
	{
		assert(task != 0);
		Instrument::taskIsReady(task->getInstrumentationTaskId());
		return _scheduler->addReadyTask(task, hardwarePlace, hint);
	}
	
	//! \brief Add back a task that was blocked but that is now unblocked
	//!
	//! \param[in] unblockedTask the task that has been unblocked
	//! \param[in] hardwarePlace the hardware place of the unblocker
	static inline void taskGetsUnblocked(Task *unblockedTask, ComputePlace *hardwarePlace)
	{
		assert(unblockedTask != 0);
		Instrument::taskIsReady(unblockedTask->getInstrumentationTaskId());
		_scheduler->taskGetsUnblocked(unblockedTask, hardwarePlace);
	}
	
	//! \brief Check if a hardware place is idle and can be resumed
	//! This call first checks if the hardware place is idle. If so, it marks it as not idle
	//! and returns true. Otherwise it returns false.
	//!
	//! \param[in] hardwarePlace the hardware place to check
	//!
	//! \returns true if the hardware place must be resumed
	static inline bool checkIfIdleAndGrantReactivation(ComputePlace *hardwarePlace)
	{
		return _scheduler->checkIfIdleAndGrantReactivation(hardwarePlace);
	}
	
	//! \brief Get a ready task for execution
	//!
	//! \param[in] hardwarePlace the hardware place asking for scheduling orders
	//! \param[in] currentTask a task within whose context the resulting task will run
	//!
	//! \returns a ready task or nullptr
	static inline Task *getReadyTask(ComputePlace *hardwarePlace, Task *currentTask = nullptr)
	{
		return _scheduler->getReadyTask(hardwarePlace, currentTask);
	}
	
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
	//! \param[in] hardwarePlace the hardware place that is about to be disabled
	static void disableComputePlace(ComputePlace *hardwarePlace)
	{
		_scheduler->disableComputePlace(hardwarePlace);
	}
	
	//! \brief Notify the scheduler that a hardware place is back online so that it preassign tasks to it
	//! 
	//! \param[in] hardwarePlace the hardware place that is about to be enabled
	static void enableComputePlace(ComputePlace *hardwarePlace)
	{
		_scheduler->enableComputePlace(hardwarePlace);
	}
	
	//! \brief Attempt to get a one task polling slot
	//! 
	//! \param[in] hardwarePlace the hardware place asking for scheduling orders
	//! \param[out] pollingSlot a pointer to a location that the caller will poll for ready tasks
	//! 
	//! \returns true if the caller is allowed to poll that memory position for a single ready task or if it actually got a task, otherwise false and the hardware place is assumed to become idle
	static inline bool requestPolling(ComputePlace *hardwarePlace, std::atomic<Task *> *pollingSlot)
	{
		return _scheduler->requestPolling(hardwarePlace, pollingSlot);
	}
	
	//! \brief Attempt to release the polling slot
	//! 
	//! \param[in] hardwarePlace the hardware place asking for scheduling orders
	//! \param[out] pollingSlot a pointer to a location that the caller is polling for ready tasks
	//! 
	//! \returns true if the caller has successfully released the polling slot otherwise false indicating that there already is a taskl assigned or it is on the way
	static bool releasePolling(ComputePlace *hardwarePlace, std::atomic<Task *> *pollingSlot)
	{
		return _scheduler->releasePolling(hardwarePlace, pollingSlot);
	}
	
};


#endif // SCHEDULER_HPP
