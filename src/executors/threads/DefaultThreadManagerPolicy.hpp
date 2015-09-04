#ifndef DEFAULT_THREAD_MANAGER_POLICY_HPP
#define DEFAULT_THREAD_MANAGER_POLICY_HPP


#include "ThreadManagerPolicyInterface.hpp"


class DefaultThreadManagerPolicy: public ThreadManagerPolicyInterface {
public:
	inline DefaultThreadManagerPolicy()
		: ThreadManagerPolicyInterface()
	{
	}
	
	virtual ~DefaultThreadManagerPolicy()
	{
	}
	
	bool checkIfMustRunInline(Task *replacementTask, Task *currentTask, CPU *cpu);
	bool checkIfUnblockedMustPreemtUnblocker(Task *unblockerTask, Task *unblockedTask, CPU *cpu);
};


#endif // DEFAULT_THREAD_MANAGER_POLICY_HPP
