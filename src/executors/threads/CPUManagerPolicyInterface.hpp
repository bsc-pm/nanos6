/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_MANAGER_POLICY_INTERFACE_HPP
#define CPU_MANAGER_POLICY_INTERFACE_HPP

#include "hardware/places/ComputePlace.hpp"


enum CPUManagerPolicyHint {
	IDLE_CANDIDATE,
	REQUEST_CPUS,
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
	//! \param[in] numRequested If hint == REQUEST_CPUS, numRequested is the amount
	//! of idle CPUs to resume
	virtual void execute(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numRequested = 0) = 0;

};


#endif // CPU_MANAGER_POLICY_INTERFACE_HPP
