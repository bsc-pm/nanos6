/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

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
