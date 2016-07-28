#ifndef SCHEDULER_HPP
#define SCHEDULER_HPP


#include "SchedulerInterface.hpp"

#include <InstrumentTaskStatus.hpp>
#include <tasks/Task.hpp>

#include <cassert>


class HardwareDescription;
class HardwarePlace;

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
	//! \returns an idle HardwarePlace that is to be resumed or nullptr
	static inline HardwarePlace *addReadyTask(Task *task, HardwarePlace *hardwarePlace, SchedulerInterface::ReadyTaskHint hint = SchedulerInterface::NO_HINT)
	{
		assert(task != 0);
		Instrument::taskIsReady(task->getInstrumentationTaskId());
		return _scheduler->addReadyTask(task, hardwarePlace, hint);
	}
	
	//! \brief Add back a task that was blocked but that is now unblocked
	//!
	//! \param[in] unblockedTask the task that has been unblocked
	//! \param[in] hardwarePlace the hardware place of the unblocker
	static inline void taskGetsUnblocked(Task *unblockedTask, HardwarePlace *hardwarePlace)
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
	static inline bool checkIfIdleAndGrantReactivation(HardwarePlace *hardwarePlace)
	{
		return _scheduler->checkIfIdleAndGrantReactivation(hardwarePlace);
	}
	
	//! \brief Get a ready task for execution
	//!
	//! \param[in] hardwarePlace the hardware place asking for scheduling orders
	//! \param[in] currentTask a task within whose context the resulting task will run
	//!
	//! \returns a ready task or nullptr
	static inline Task *getReadyTask(HardwarePlace *hardwarePlace, Task *currentTask = nullptr)
	{
		return _scheduler->getReadyTask(hardwarePlace, currentTask);
	}
	
	//! \brief Get an idle hardware place
	//!
	//! \param[in] force idicates that an idle hardware place must be returned (if any) even if the scheduler does not have any pending work to be assigned
	//!
	//! \returns a hardware place that becomes non idle or nullptr
	static inline HardwarePlace *getIdleHardwarePlace(bool force=false)
	{
		return _scheduler->getIdleHardwarePlace(force);
	}
	
};


#endif // SCHEDULER_HPP
