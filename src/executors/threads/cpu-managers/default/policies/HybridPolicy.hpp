/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef HYBRID_POLICY_HPP
#define HYBRID_POLICY_HPP

#include "IdlePolicy.hpp"
#include "executors/threads/CPUManagerPolicyInterface.hpp"
#include "executors/threads/cpu-managers/default/DefaultCPUManager.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "support/config/ConfigVariable.hpp"


class HybridPolicy : public IdlePolicy {
private:
	//! The maximum number of iterations to wait before assigning a null task.
	//! This variable refers to the sum of iterations for all CPUs, thus to
	//! obtain the number per CPU it must be divided by _numCPUs
	ConfigVariable<size_t> _numBusyIters;

public:
	inline HybridPolicy(DefaultCPUManager &cpuManager, size_t numCPUs) :
		IdlePolicy(cpuManager, numCPUs),
		_numBusyIters("cpumanager.busy_iters")
	{
	}

	inline size_t getMaxBusyIterations() const
	{
		// In the busy policy, the maximum number of busy iterations should
		// be a value large enough so that it can barely be reached
		return ((size_t) (_numBusyIters.getValue() / _numCPUs));
	}
};

#endif // HYBRID_POLICY_HPP
