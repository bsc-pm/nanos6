/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_MANAGER_POLICY_INTERFACE_HPP
#define CPU_MANAGER_POLICY_INTERFACE_HPP

#include "hardware/places/ComputePlace.hpp"


enum CPUManagerPolicyHint {
	IDLE_CANDIDATE,
	ADDED_TASKS,
	HANDLE_TASKFOR
};


class CPUManagerPolicyInterface {

public:

	virtual inline ~CPUManagerPolicyInterface()
	{
	}

	//! \brief Execute the CPUManager's policy
	//!
	//! \param[in,out] cpu The CPU that triggered the call, if any
	//! \param[in] hint A hint about what kind of change triggered this call
	//! \param[in] numTasks If hint == ADDED_TASKS, numTasks is the amount
	//! of tasks added
	virtual void execute(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numTasks = 0) = 0;

};


#endif // CPU_MANAGER_POLICY_INTERFACE_HPP
