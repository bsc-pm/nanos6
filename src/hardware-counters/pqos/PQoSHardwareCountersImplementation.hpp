/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PQOS_HARDWARE_COUNTERS_IMPLEMENTATION_HPP
#define PQOS_HARDWARE_COUNTERS_IMPLEMENTATION_HPP

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "hardware-counters/HardwareCountersInterface.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "tasks/Task.hpp"


namespace Boost = boost::accumulators;
namespace BoostTag = boost::accumulators::tag;

class PQoSHardwareCountersImplementation : public HardwareCountersInterface {

private:

	enum supported_counters_t {
		pqos_llc_usage = 0,
		pqos_ipc,
		pqos_local_mem_bandwidth,
		pqos_remote_mem_bandwidth,
		pqos_llc_miss_rate,
		num_pqos_counters
	};

	typedef Boost::accumulator_set<double, Boost::stats<BoostTag::sum, BoostTag::mean, BoostTag::variance, BoostTag::count> > statistics_accumulator_t;
	typedef std::map<std::string, std::vector<statistics_accumulator_t> > statistics_map_t;

	//! A map of HW counter statistics per tasktype
	static statistics_map_t _statistics;

	//! Ensures atomic access to the tasktype map
	static SpinLock _statsLock;

	//! Whether PQoS HW Counter instrumentation is enabled
	static bool _enabled;

	//! Whether the verbose mode is activated
	static bool _verbose;

	//! The file name on which to output statistics when verbose is enabled
	static std::string _verboseFile;

private:

	inline void displayStatistics()
	{
		// Try opening the output file
		std::ios_base::openmode openMode = std::ios::out;
		std::ofstream output(_verboseFile, openMode);
		FatalErrorHandler::warnIf(
			!output.is_open(),
			"Could not create or open the verbose file: ", _verboseFile, ". ",
			"Using standard output."
		);

		// Retrieve statistics
		std::stringstream outputStream;
		outputStream << std::left << std::fixed << std::setprecision(5);
		outputStream << "-------------------------------\n";

		// Iterate through all tasktypes
		for (auto &it : _statistics) {
			size_t instances = Boost::count(it.second[0]);
			if (instances > 0) {
				std::string typeLabel = it.first + " (" + std::to_string(instances) + ")";

				outputStream <<
					std::setw(7)  << "STATS"                 << " " <<
					std::setw(6)  << "PQOS"                  << " " <<
					std::setw(39) << "TASK-TYPE (INSTANCES)" << " " <<
					std::setw(30) << typeLabel               << "\n";

				// Iterate through all counter types
				for (unsigned short id = 0; id < num_pqos_counters; ++id) {
					double counterAvg   = Boost::mean(it.second[id]);
					double counterStdev = sqrt(Boost::variance(it.second[id]));
					double counterSum   = Boost::sum(it.second[id]);

					// In KB
					if (id == HWCounters::llc_usage ||
						id == HWCounters::local_mem_bandwidth ||
						id == HWCounters::remote_mem_bandwidth
					) {
						counterAvg   /= 1024.0;
						counterStdev /= 1024.0;
						counterSum   /= 1024.0;
					}

					outputStream <<
						std::setw(7)  << "STATS"                             << " "   <<
						std::setw(6)  << "PQOS"                              << " "   <<
						std::setw(39) << HWCounters::counterDescriptions[id] << " "   <<
						std::setw(30) << "SUM / AVG / STDEV"                 << " "   <<
						std::setw(15) << counterSum                          << " / " <<
						std::setw(15) << counterAvg                          << " / " <<
						std::setw(15) << counterStdev                        << "\n";
				}
				outputStream << "-------------------------------\n";
			}
		}

		if (output.is_open()) {
			// Output into the file and close it
			output << outputStream.str();
			output.close();
		} else {
			std::cout << outputStream.str();
		}
	}

public:

	void initialize(bool verbose, std::string verboseFile);

	void shutdown();

	inline bool isSupported(HWCounters::counters_t counterType)
	{
		if (counterType == HWCounters::llc_usage ||
			counterType == HWCounters::ipc ||
			counterType == HWCounters::local_mem_bandwidth ||
			counterType == HWCounters::remote_mem_bandwidth ||
			counterType == HWCounters::llc_miss_rate
		) {
			// pqos_llc_usage == HWCounters::llc_usage
			// pqos_ipc == HWCounters::ipc
			// pqos_local_mem_bandwidth == HWCounters::local_mem_bandwidth
			// pqos_remote_mem_bandwidth == HWCounters::remote_mem_bandwidth
			// pqos_llc_miss_rate == HWCounters::llc_miss_rate
			return true;
		} else {
			return false;
		}
	}

	void threadInitialized();

	void threadShutdown();

	void taskCreated(Task *task, bool enabled);

	void taskReinitialized(Task *task);

	void taskStarted(Task *task);

	void taskStopped(Task *task);

	void taskFinished(Task *task);

	size_t getTaskHardwareCountersSize() const;

};

#endif // PQOS_HARDWARE_COUNTERS_IMPLEMENTATION_HPP
