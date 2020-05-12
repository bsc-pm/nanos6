/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef BUSY_POLICY_HPP
#define BUSY_POLICY_HPP

#include "executors/threads/CPUManagerPolicyInterface.hpp"

#include "InstrumentWorkerThread.hpp"

class ComputePlace;


class BusyPolicy : public CPUManagerPolicyInterface {

public:

	inline void execute(ComputePlace *, CPUManagerPolicyHint hint, size_t = 0)
	{
		// NOTE: This policy works as follows:
		// - If the hint is IDLE_CANDIDATE, the CPU remains active (no change)
		// - If the hint is REQUEST_CPUS or HANDLE_TASKFOR, no CPUs are woken up as
		//   all of them should be awake

		if (hint == IDLE_CANDIDATE)
			Instrument::WorkerBusyWaits();
	}

};

#endif // BUSY_POLICY_HPP
