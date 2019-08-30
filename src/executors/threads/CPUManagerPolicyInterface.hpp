/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_MANAGER_POLICY_INTERFACE_HPP
#define CPU_MANAGER_POLICY_INTERFACE_HPP

#include "CPU.hpp"


enum CPUManagerPolicyHint {
	IDLE_CANDIDATE,
	ADDED_TASKS
};


class CPUManagerPolicyInterface {

public:
	
	virtual ~CPUManagerPolicyInterface()
	{
	}
	
	virtual void executePolicy(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numTasks) = 0;
	
};

#endif // CPU_MANAGER_POLICY_INTERFACE_HPP
