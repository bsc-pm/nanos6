/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef HYBRID_POLICY_HPP
#define HYBRID_POLICY_HPP

#include "IdlePolicy.hpp"
#include "executors/threads/CPUManagerPolicyInterface.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "support/config/ConfigVariable.hpp"


class HybridPolicy : public CPUManagerPolicyInterface {

private:

	//! The maximum amount of CPUs in the system
	size_t _numCPUs;

	//! The maximum number of iterations to wait before assigning a null task
	static ConfigVariable<size_t> _numBusyIters;

public:

	inline HybridPolicy(size_t numCPUs)
		: _numCPUs(numCPUs)
	{
	}

	inline void execute(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numRequested = 0)
	{
		// The fault behavior is the same as the idle policy
		IdlePolicy::idlePolicyDefaultExecution(cpu, hint, numRequested, _numCPUs);
	}

	inline size_t getMaxBusyIterations() const
	{
		// In the busy policy, the maximum number of busy iterations should
		// be a value large enough so that it can barely be reached
		return _numBusyIters.getValue();
	}
};

#endif // HYBRID_POLICY_HPP
