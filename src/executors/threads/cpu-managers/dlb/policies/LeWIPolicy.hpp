/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef LEWI_POLICY_HPP
#define LEWI_POLICY_HPP

#include "executors/threads/CPUManagerPolicyInterface.hpp"
#include "hardware/places/ComputePlace.hpp"


class LeWIPolicy : public CPUManagerPolicyInterface {

private:

	//! The maximum amount of CPUs in the system
	size_t _numCPUs;

public:

	inline LeWIPolicy(size_t numCPUs)
		: _numCPUs(numCPUs)
	{
	}

	void execute(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numRequested = 0) override;

};

#endif // LEWI_POLICY_HPP
