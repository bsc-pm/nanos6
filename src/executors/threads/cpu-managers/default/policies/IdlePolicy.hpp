/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef IDLE_POLICY_HPP
#define IDLE_POLICY_HPP

#include "executors/threads/CPUManagerPolicyInterface.hpp"
#include "hardware/places/ComputePlace.hpp"


class IdlePolicy : public CPUManagerPolicyInterface {

private:

	//! The maximum amount of CPUs in the system
	size_t _numCPUs;

public:

	inline IdlePolicy(size_t numCPUs)
		: _numCPUs(numCPUs)
	{
	}

	void execute(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numRequested = 0);

};

#endif // IDLE_POLICY_HPP
