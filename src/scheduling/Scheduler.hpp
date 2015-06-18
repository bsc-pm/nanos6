#ifndef SCHEDULER_HPP
#define SCHEDULER_HPP


#include "SchedulerInterface.hpp"

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
	
	//! \brief Add the main task
	//!
	//! \param in mainTask the main task
	static inline void addMainTask(Task *mainTask)
	{
		_scheduler->addMainTask(mainTask);
	}
	
	//! \brief Add a (ready) task that has been freed by another task
	//!
	//! \param in newReadyTask the task to be added
	//! \param in triggererTask the task that has been finished and thus has triggered the change of status of the new task to ready
	//! \param in hardwarePlace the hardware place where the triggerer has run
	static inline void addSiblingTask(Task *newReadyTask, Task *triggererTask, HardwarePlace const *hardwarePlace)
	{
		_scheduler->addSiblingTask(newReadyTask, triggererTask, hardwarePlace);
	}
	//! \brief Add a child task that is ready
	//!
	//! \param in newReadyTask the task to be added
	//! \param in hardwarePlace the hardware place where the parent task is running
	static inline void addChildTask(Task *newReadyTask, HardwarePlace const *hardwarePlace)
	{
		_scheduler->addChildTask(newReadyTask, hardwarePlace);
	}
	
	
	//! \brief Get a ready task for execution
	//!
	//! \param in hardwarePlace the hardware place asking for scheduling orders
	//!
	//! \returns a ready task or nullptr
	static Task *schedule(HardwarePlace const *hardwarePlace)
	{
		return _scheduler->schedule(hardwarePlace);
	}
	
};


#endif // SCHEDULER_HPP
