/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PAPI_HARDWARE_COUNTERS_HPP
#define PAPI_HARDWARE_COUNTERS_HPP

#include "hardware-counters/HardwareCountersInterface.hpp"
#include "hardware-counters/SupportedHardwareCounters.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


class Task;

class PAPIHardwareCounters : public HardwareCountersInterface {

public:

	inline PAPIHardwareCounters(bool, const std::string &, const std::vector<HWCounters::counters_t> &)
	{
		FatalErrorHandler::fail("PAPI backend not supported yet");
	}

	inline ~PAPIHardwareCounters()
	{
	}

	inline void cpuBecomesIdle(CPUHardwareCountersInterface *, ThreadHardwareCountersInterface *) override
	{
	}

	inline void threadInitialized(ThreadHardwareCountersInterface *) override
	{
	}

	inline void threadShutdown(ThreadHardwareCountersInterface *) override
	{
	}

	inline void taskReinitialized(TaskHardwareCountersInterface *) override
	{
	}

	inline void taskStarted(
		CPUHardwareCountersInterface *,
		ThreadHardwareCountersInterface *,
		TaskHardwareCountersInterface *
	) override {
	}

	inline void taskStopped(
		CPUHardwareCountersInterface *,
		ThreadHardwareCountersInterface *,
		TaskHardwareCountersInterface *
	) override {
	}

};

#endif // PAPI_HARDWARE_COUNTERS_HPP
