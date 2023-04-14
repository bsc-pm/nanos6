/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef IDLE_POLICY_HPP
#define IDLE_POLICY_HPP

#include "executors/threads/CPUManagerPolicyInterface.hpp"
#include "executors/threads/cpu-managers/default/DefaultCPUManager.hpp"
#include "hardware/places/ComputePlace.hpp"


class IdlePolicy : public CPUManagerPolicyInterface {
protected:
	//! Reference to the default CPU manager
	DefaultCPUManager &_cpuManager;

	//! The maximum amount of CPUs in the system
	size_t _numCPUs;

public:
	inline IdlePolicy(DefaultCPUManager &cpuManager, size_t numCPUs) :
		_cpuManager(cpuManager),
		_numCPUs(numCPUs)
	{
	}

	void execute(
		ComputePlace *cpu,
		CPUManagerPolicyHint hint,
		size_t numRequested = 0
	);
};

#endif // IDLE_POLICY_HPP
