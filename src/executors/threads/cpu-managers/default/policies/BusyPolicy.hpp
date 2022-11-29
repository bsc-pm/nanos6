/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef BUSY_POLICY_HPP
#define BUSY_POLICY_HPP

#include <limits>

#include "InstrumentWorkerThread.hpp"
#include "executors/threads/CPUManagerPolicyInterface.hpp"


class ComputePlace;


class BusyPolicy : public CPUManagerPolicyInterface {

public:

	inline void execute(ComputePlace *, CPUManagerPolicyHint hint, size_t = 0)
	{
		// NOTE: This policy works as follows:
		// - If the hint is IDLE_CANDIDATE, the CPU remains active (no change)
		// - If the hint is REQUEST_CPUS, no CPUs are woken up as all of them should be awake

		if (hint == IDLE_CANDIDATE)
			Instrument::workerThreadBusyWaits();
	}

	inline size_t getMaxBusyIterations() const
	{
		// In the busy policy, the maximum number of busy iterations should
		// be a value large enough so that it can barely be reached
		return std::numeric_limits<std::size_t>::max();
	}

};

#endif // BUSY_POLICY_HPP
