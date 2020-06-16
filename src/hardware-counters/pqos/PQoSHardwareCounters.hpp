/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PQOS_HARDWARE_COUNTERS_HPP
#define PQOS_HARDWARE_COUNTERS_HPP

#include <pqos.h>
#include <vector>

#include "hardware-counters/CPUHardwareCountersInterface.hpp"
#include "hardware-counters/HardwareCountersInterface.hpp"
#include "hardware-counters/TaskHardwareCountersInterface.hpp"
#include "hardware-counters/ThreadHardwareCountersInterface.hpp"


class PQoSHardwareCounters : public HardwareCountersInterface {

private:

	//! Whether PQoS HW Counter instrumentation is enabled
	bool _enabled;

	//! An enumeration containing the events that we monitor
	enum pqos_mon_event _monitoredEvents;

	//! The number of enabled counters (enabled by the user and available)
	size_t _numEnabledCounters;

public:

	//! \brief Initializ the PQoS backend
	//!
	//! \param[in] verbose Whether verbose mode is enabled
	//! \param[in] verboseFile The file onto which to write verbose messages
	//! \param[in,out] enabledEvents A vector with all the events enabled by the user,
	//! which will be modified to disable those that are unavailable
	//! \param[in,out] eventMap A map with every possible event and a bool indicating
	//! whether it is enabled or not. Those that are enabled but not available will
	//! be turned into false
	PQoSHardwareCounters(
		bool,
		const std::string &,
		std::vector<HWCounters::counters_t> &enabledEvents,
		std::map<HWCounters::counters_t, bool> &eventMap
	);

	~PQoSHardwareCounters();

	inline size_t getNumEnabledCounters() const override
	{
		return _numEnabledCounters;
	}

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
