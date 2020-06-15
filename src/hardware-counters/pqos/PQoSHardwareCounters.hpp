/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PQOS_HARDWARE_COUNTERS_HPP
#define PQOS_HARDWARE_COUNTERS_HPP

#include "hardware-counters/CPUHardwareCountersInterface.hpp"
#include "hardware-counters/HardwareCountersInterface.hpp"
#include "hardware-counters/TaskHardwareCountersInterface.hpp"
#include "hardware-counters/ThreadHardwareCountersInterface.hpp"
#include "tasks/Task.hpp"


class PQoSHardwareCounters : public HardwareCountersInterface {

private:

	//! Whether PQoS HW Counter instrumentation is enabled
	bool _enabled;

	//! An enumeration containing the events that we monitor
	enum pqos_mon_event _monitoredEvents;

public:

	PQoSHardwareCounters(bool, const std::string &, std::vector<HWCounters::counters_t> &enabledEvents);

	~PQoSHardwareCounters();

	void threadInitialized(ThreadHardwareCountersInterface *threadCounters) override;

	void threadShutdown(ThreadHardwareCountersInterface *threadCounters) override;

	void taskReinitialized(TaskHardwareCountersInterface *taskCounters) override;

	void readTaskCounters(
		ThreadHardwareCountersInterface *threadCounters,
		TaskHardwareCountersInterface *taskCounters
	) override;

	void readCPUCounters(
		CPUHardwareCountersInterface *cpuCounters,
		ThreadHardwareCountersInterface *threadCounters
	) override;

};

#endif // PQOS_HARDWARE_COUNTERS_HPP
