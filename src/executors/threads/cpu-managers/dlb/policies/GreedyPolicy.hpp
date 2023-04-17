/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef GREEDY_POLICY_HPP
#define GREEDY_POLICY_HPP

#include "executors/threads/CPUManagerPolicyInterface.hpp"
#include "hardware/places/ComputePlace.hpp"


class GreedyPolicy : public CPUManagerPolicyInterface {

private:

	//! The maximum amount of CPUs in the system
	size_t _numCPUs;

public:

	inline GreedyPolicy(size_t numCPUs)
		: _numCPUs(numCPUs)
	{
	}

	void execute(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numRequested = 0) override;

};

#endif // GREEDY_POLICY_HPP
