/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef PQOS_TASK_HARDWARE_COUNTERS_MONITOR_HPP
#define PQOS_TASK_HARDWARE_COUNTERS_MONITOR_HPP

#include <iomanip>
#include <iostream>
#include <map>
#include <pqos.h>
#include <sstream>

#include "TasktypeHardwareCountersPredictions.hpp"


class TaskHardwareCountersMonitor {

private:
	
	typedef std::map< std::string, TasktypeHardwareCountersPredictions *> tasktype_hardware_counters_map_t;
	
	//! The singleton instance
	static TaskHardwareCountersMonitor *_monitor;
	
	//! Maps hw counter predictions per tasktype
	tasktype_hardware_counters_map_t _tasktypeMap;
	
	//! Ensures atomic access to the tasktype map
	SpinLock _spinlock;
	
	
private:
	
	inline TaskHardwareCountersMonitor() :
		_tasktypeMap(),
		_spinlock()
	{
	}
	
	
public:
	
	// Delete copy and move constructors/assign operators
	TaskHardwareCountersMonitor(TaskHardwareCountersMonitor const&) = delete;            // Copy construct
	TaskHardwareCountersMonitor(TaskHardwareCountersMonitor&&) = delete;                 // Move construct
	TaskHardwareCountersMonitor& operator=(TaskHardwareCountersMonitor const&) = delete; // Copy assign
	TaskHardwareCountersMonitor& operator=(TaskHardwareCountersMonitor &&) = delete;     // Move assign
	
	
	//! \brief Initialization of the task hardware counter monitor
	static inline void initialize()
	{
		// Create the monitoring module
		if (_monitor == nullptr) {
			_monitor = new TaskHardwareCountersMonitor();
		}
	}
	
	//! \brief Shutdown of the task hardware counter monitor
	static inline void shutdown()
	{
		if (_monitor != nullptr) {
			// Destroy all the tasktype hardware counter structures
			for (auto &it : _monitor->_tasktypeMap) {
				if (it.second != nullptr) {
					delete it.second;
				}
			}
			
			// Destroy the hw counter monitoring module
			delete _monitor;
		}
	}
	
	//! \brief Display task hardware counter statistics
	//! \param output The output stream
	static inline void displayStatistics(std::stringstream &output)
	{
		if (_monitor != nullptr) {
			// Number of different types of tasks
			// Averages of hardware counters including all tasktypes
			// Total sums of hardware counters including all tasks
			short numTasktypes = 0;
			std::vector<double> avgPerCounter(HWCounters::num_counters, 0.0);
			std::vector<double> sumPerCounter(HWCounters::num_counters, 0.0);
			
			output << std::left << std::fixed << std::setprecision(5);
			output << "-------------------------------\n";
			
			// Iterate through all tasktypes of task
			for (auto &it : _monitor->_tasktypeMap) {
				++numTasktypes;
				size_t instances = it.second->getInstances();
				std::string typeLabel = it.first + " (" + std::to_string(instances) + ")";
				
				output <<
					std::setw(7)  << "STATS"                 << " " <<
					std::setw(6)  << "PQOS"                  << " " <<
					std::setw(39) << "TASK-TYPE (INSTANCES)" << " " <<
					std::setw(30) << typeLabel               << "\n";
				
				// Iterate through all counter types
				for (unsigned short i = 0; i < HWCounters::num_counters; ++i) {
					double counterAvg   = it.second->getCounterAverage((HWCounters::counters_t) i);
					double counterStdev = it.second->getCounterStdev((HWCounters::counters_t) i);
					double counterSum   = it.second->getCounterSum((HWCounters::counters_t) i);
					double accuracy     = it.second->getAverageAccuracy((HWCounters::counters_t) i);
					std::string accur = "NA";
					
					// Make sure there was at least one prediction
					if (!std::isnan(accuracy)) {
						std::ostringstream oss;
						oss << std::setprecision(2) << std::fixed << accuracy;
						accur = oss.str() + "%";
					}
					
					// In KB
					if (i == HWCounters::llc_usage            ||
						i == HWCounters::local_mem_bandwidth  ||
						i == HWCounters::remote_mem_bandwidth
					) {
						counterAvg   /= 1024.0;
						counterStdev /= 1024.0;
						counterSum   /= 1024.0;
					}
					avgPerCounter[i] += counterAvg;
					sumPerCounter[i] += counterSum;
					
					output <<
						std::setw(7)  << "STATS"                            << " "   <<
						std::setw(6)  << "PQOS"                             << " "   <<
						std::setw(39) << HWCounters::counterDescriptions[i] << " "   <<
						std::setw(30) << "ACCURACY / SUM / AVG / STDEV"     << " "   <<
						std::setw(10) << std::right << accur << std::left   << " / " <<
						std::setw(15) << counterSum                         << " / " <<
						std::setw(15) << counterAvg                         << " / " <<
						std::setw(15) << counterStdev                       << "\n";
				}
				output << "-------------------------------\n";
			}
			
			// Print statistics associated to all tasks
			output <<
				std::setw(7)  << "STATS"     << " " <<
				std::setw(6)  << "PQOS"      << " " <<
				std::setw(39) << "ALL TASKS" << "\n";
			
			for (unsigned short i = 0; i < HWCounters::num_counters; ++i) {
				output <<
					std::setw(7)  << "STATS"                            << " "   <<
					std::setw(6)  << "PQOS"                             << " "   <<
					std::setw(39) << HWCounters::counterDescriptions[i] << " "   <<
					std::setw(12) << "SUM / AVG"                        << " "   <<
					std::setw(15) << sumPerCounter[i]                   << " / " <<
					std::setw(15) << avgPerCounter[i]/numTasktypes      << "\n";
			}
			
			output << "-------------------------------\n";
		}
	}
	
	//! \brief Gather information about a task
	//! \param taskCounters The task's hardware counter structures
	//! \param label The tasktype
	//! \param cost The task's computational cost
	static void taskCreated(TaskHardwareCounters *taskCounters, const std::string &label, size_t cost);
	
	//! \brief Predict hardware counter values for a task
	//! \param taskPredictions The task's hardware counter prediction structures
	//! \param label The tasktype
	//! \param cost The task's computational cost
	static void predictTaskCounters(TaskHardwareCountersPredictions *taskPredictions, const std::string &label, size_t cost);
	
	//! \brief Start hardware counter monitoring for a task
	//! \param taskCounters The task's hardware counter structures
	//! \param threadData The monitoring data of the thread executing the task
	static void startTaskMonitoring(TaskHardwareCounters *taskCounters, pqos_mon_data *threadData);
	
	//! \brief Stop hardware counter monitoring for a task
	//! \param taskCounters The task's hardware counter structures
	//! \param threadData The monitoring data of the thread executing the task
	static void stopTaskMonitoring(TaskHardwareCounters *taskCounters, pqos_mon_data *threadData);
	
	//! \brief Finish hardware counter monitoring for a task and accumulate
	//! the counters into its tasktype's counters
	//! \param taskCounters The task's hardware counter structures
	//! \param threadData The monitoring data of the thread executing the task
	static void taskFinished(TaskHardwareCounters *taskCounters, TaskHardwareCountersPredictions *taskPredictions);
	
	//! \brief Insert normalized counter values (values per unit of cost)
	//! \param label The tasktype
	//! \param counterIds A vector of counter identifiers
	//! \param counterValues A vector of normalized counter values
	static void insertCounterValuesPerUnitOfCost(
		const std::string &label,
		std::vector<HWCounters::counters_t> &counterIds,
		std::vector<double> &counterValues
	);
	
	//! \brief Get the average counter values per unit of cost of all the
	//! tasktypes being monitored
	//! \param[out] labels The reference of a vector in which all the available
	//! tasktypes will be inserted
	//! \param[out] unitaryValues The reference of a vector in which the respective
	//! vectors of counter identifiers and counter values will be inserted
	static void getAverageCounterValuesPerUnitOfCost(
		std::vector<std::string> &labels,
		std::vector<std::vector<std::pair<HWCounters::counters_t, double>>> &unitaryValues
	);
	
};

#endif // PQOS_TASK_HARDWARE_COUNTERS_MONITOR_HPP
