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

#include "PQoSTaskHardwareCounters.hpp"
#include "hardware-counters/HardwareCountersInterface.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
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

private:

	void displayStatistics();

public:

	PQoSHardwareCounters(bool verbose, const std::string &verboseFile);

	~PQoSHardwareCounters();

	void threadInitialized();

	void threadShutdown();

	void taskCreated(Task *task, bool enabled);

	void taskReinitialized(Task *task);

	void taskStarted(Task *task);

	void taskStopped(Task *task);

	void taskFinished(Task *task);

};

#endif // PQOS_HARDWARE_COUNTERS_HPP
