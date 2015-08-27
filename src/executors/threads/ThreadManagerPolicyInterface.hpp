#ifndef THREAD_MANAGER_POLICY_INTERFACE_HPP
#define THREAD_MANAGER_POLICY_INTERFACE_HPP


class CPU;
class Task;


class ThreadManagerPolicyInterface {
public:
	virtual ~ThreadManagerPolicyInterface()
	{
	}
	
	//! \brief check if a task must be run within the thread of a blocked task
	//!
	//! \param[in] replacementTask the task that is to be run
	//! \param[in] currentTask the task that is blocked on the thread
	//! \param[in] cpu the CPU where currentTask is runnun
	//!
	//! \returns true is replacementTask is to be run within the same thread as currentTask (which is blocked)
	virtual bool checkIfMustRunInline(Task *replacementTask, Task *currentTask, CPU *cpu) = 0;
	
};


#endif // THREAD_MANAGER_POLICY_INTERFACE_HPP
