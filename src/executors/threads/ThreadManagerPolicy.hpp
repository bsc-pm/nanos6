#ifndef THREAD_MANAGER_POLICY_HPP
#define THREAD_MANAGER_POLICY_HPP

#include <cassert>

#include "tasks/Task.hpp"

#include "ThreadManagerPolicyInterface.hpp"

//! \brief This class is the main interface within the runtime to interact with the thread management policy
//!
//! It holds a pointer to the actual policy and forwards the calls to it.
class ThreadManagerPolicy {
	static ThreadManagerPolicyInterface *_policy;
	
public:
	//! \brief Initializes the _scheduler member and in turn calls its initialization method
	static void initialize();
	
	//! \brief check if a task must be run within the thread of a blocked task
	//!
	//! \param[in] replacementTask the task that is to be run
	//! \param[in] currentTask the task that is blocked on the thread
	//! \param[in] cpu the CPU where currentTask is running
	//!
	//! \returns true is replacementTask is to be run within the same thread as currentTask (which is blocked)
	static bool checkIfMustRunInline(Task *replacementTask, Task *currentTask, CPU *cpu)
	{
		assert(_policy != nullptr);
		assert(replacementTask != nullptr);
		assert(currentTask != nullptr);
		assert(cpu != nullptr);
		
		if (replacementTask->getThread() != nullptr) {
			return false;
		}
		
		return _policy->checkIfMustRunInline(replacementTask, currentTask, cpu);
	}
};


#endif // THREAD_MANAGER_POLICY_HPP
