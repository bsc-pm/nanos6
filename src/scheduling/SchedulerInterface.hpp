#ifndef SCHEDULER_INTERFACE_HPP
#define SCHEDULER_INTERFACE_HPP


class HardwarePlace;
class Task;


//! \brief Interface that schedulers must implement
class SchedulerInterface {
public:
	virtual ~SchedulerInterface()
	{
	}
	
	
	//! \brief Add the main task
	//!
	//! \param in mainTask the main task
	virtual void addMainTask(Task *mainTask) = 0;
	
	//! \brief Add a (ready) task that has been freed by another task
	//!
	//! \param in newReadyTask the task to be added
	//! \param in triggererTask the task that has been finished and thus has triggered the change of status of the new task to ready
	//! \param in hardwarePlace the hardware place where the triggerer has run
	virtual void addSiblingTask(Task *newReadyTask, Task *triggererTask, HardwarePlace const *hardwarePlace) = 0;
	
	//! \brief Add a child task that is ready
	//!
	//! \param in newReadyTask the task to be added
	//! \param in hardwarePlace the hardware place where the parent task is running
	virtual void addChildTask(Task *newReadyTask, HardwarePlace const *hardwarePlace) = 0;
	
	
	//! \brief Get a ready task for execution
	//!
	//! \param in hardwarePlace the hardware place asking for scheduling orders
	//!
	//! \returns a ready task or nullptr
	virtual Task *schedule(HardwarePlace const *hardwarePlace) = 0;
	
};


#endif // SCHEDULER_INTERFACE_HPP
