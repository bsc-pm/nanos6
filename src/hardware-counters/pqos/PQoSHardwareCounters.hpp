/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PQOS_HARDWARE_COUNTERS_HPP
#define PQOS_HARDWARE_COUNTERS_HPP

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "hardware-counters/HardwareCountersInterface.hpp"
#include "hardware-counters/TaskHardwareCountersInterface.hpp"
#include "hardware-counters/ThreadHardwareCountersInterface.hpp"
#include "tasks/Task.hpp"


class PQoSHardwareCounters : public HardwareCountersInterface {

private:

	typedef BoostAcc::accumulator_set<double, BoostAcc::stats<BoostAccTag::sum, BoostAccTag::mean, BoostAccTag::variance, BoostAccTag::count> > statistics_accumulator_t;
	typedef std::map<std::string, std::vector<statistics_accumulator_t> > statistics_map_t;

	//! A map of HW counter statistics per tasktype
	statistics_map_t _statistics;

	//! Ensures atomic access to the tasktype map
	SpinLock _statsLock;

	//! Whether PQoS HW Counter instrumentation is enabled
	bool _enabled;

	//! Whether the verbose mode is activated
	bool _verbose;

	//! The file name on which to output statistics when verbose is enabled
	std::string _verboseFile;

	//! An enumeration containing the events that we monitor
	enum pqos_mon_event _monitoredEvents;

	bool _enabledEvents[HWCounters::PQOS_NUM_EVENTS];

private:

	void displayStatistics();

public:

	PQoSHardwareCounters(bool verbose, const std::string &verboseFile, const std::vector<bool> &enabledEvents);

	~PQoSHardwareCounters();

	void cpuBecomesIdle(
		CPUHardwareCountersInterface *cpuCounters,
		ThreadHardwareCountersInterface *threadCounters
	);

	void threadInitialized(ThreadHardwareCountersInterface *threadCounters);

	void threadShutdown(ThreadHardwareCountersInterface *threadCounters);

	void taskCreated(Task *task, bool enabled);

	void taskReinitialized(TaskHardwareCountersInterface *taskCounters);

	void taskStarted(
		CPUHardwareCountersInterface *cpuCounters,
		ThreadHardwareCountersInterface *threadCounters,
		TaskHardwareCountersInterface *taskCounters
	);

	void taskStopped(
		CPUHardwareCountersInterface *cpuCounters,
		ThreadHardwareCountersInterface *threadCounters,
		TaskHardwareCountersInterface *taskCounters
	);

	void taskFinished(Task *task, TaskHardwareCountersInterface *taskCounters);

};

#endif // PQOS_HARDWARE_COUNTERS_HPP
