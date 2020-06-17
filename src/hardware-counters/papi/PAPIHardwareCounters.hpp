/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PAPI_HARDWARE_COUNTERS_HPP
#define PAPI_HARDWARE_COUNTERS_HPP

#include "hardware-counters/CPUHardwareCountersInterface.hpp"
#include "hardware-counters/HardwareCountersInterface.hpp"
#include "hardware-counters/SupportedHardwareCounters.hpp"
#include "hardware-counters/TaskHardwareCountersInterface.hpp"
#include "hardware-counters/ThreadHardwareCountersInterface.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


class PAPIHardwareCounters : public HardwareCountersInterface {

public:

	inline PAPIHardwareCounters(
		bool,
		const std::string &,
		std::vector<HWCounters::counters_t> &
	) {
		FatalErrorHandler::fail("PAPI backend not supported yet");
	}

	inline ~PAPIHardwareCounters()
	{
	}

	static inline size_t getNumEnabledCounters() const
	{
		return 0;
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

	inline void readTaskCounters(
		ThreadHardwareCountersInterface *,
		TaskHardwareCountersInterface *
	) override {
	}

	inline void readCPUCounters(
		CPUHardwareCountersInterface *,
		ThreadHardwareCountersInterface *
	) override {
	}

};

#endif // PAPI_HARDWARE_COUNTERS_HPP
