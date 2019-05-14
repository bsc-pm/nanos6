/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_MONITOR_HPP
#define TASK_MONITOR_HPP

#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "TasktypePredictions.hpp"
#include "lowlevel/SpinLock.hpp"


class TaskMonitor {

private:
	
	typedef std::map< std::string, TasktypePredictions *> tasktype_map_t;
	
	// The monitor singleton instance
	static TaskMonitor *_monitor;
	
	//! Maps TasktypePredictions by task labels
	tasktype_map_t _tasktypeMap;
	
	//! Spinlock that ensures atomic access within the tasktype map
	SpinLock _spinlock;
	
	
private:
	
	inline TaskMonitor() :
		_tasktypeMap(),
		_spinlock()
	{
	}
	
	
public:
	
	// Delete copy and move constructors/assign operators
	TaskMonitor(TaskMonitor const&) = delete;            // Copy construct
	TaskMonitor(TaskMonitor&&) = delete;                 // Move construct
	TaskMonitor& operator=(TaskMonitor const&) = delete; // Copy assign
	TaskMonitor& operator=(TaskMonitor &&) = delete;     // Move assign
	
	
	//! \brief Initialize task monitoring
	static inline void initialize()
	{
		// Create the monitoring module
		if (_monitor == nullptr) {
			_monitor = new TaskMonitor();
		}
	}
	
	//! \brief Shutdown task monitoring
	static inline void shutdown()
	{
		if (_monitor != nullptr) {
			// Destroy all the task type statistics
			for (auto &it : _monitor->_tasktypeMap) {
				if (it.second != nullptr) {
					delete it.second;
				}
			}
			
			delete _monitor;
		}
	}
	
	//! \brief Display task statistics
	//! \param stream The output stream
	static inline void displayStatistics(std::stringstream &stream)
	{
		if (_monitor != nullptr) {
			stream << std::left << std::fixed << std::setprecision(5) << "\n";
			stream << "+-----------------------------+\n";
			stream << "|       TASK STATISTICS       |\n";
			stream << "+-----------------------------+\n";
			
			for (auto const &it : _monitor->_tasktypeMap) {
				int instances = it.second->getInstances();
				if (instances) {
					double avgCost        = it.second->getAverageTimePerUnitOfCost();
					double stdevCost      = it.second->getStdevTimePerUnitOfCost();
					double accuracy       = it.second->getPredictionAccuracy();
					std::string typeLabel = it.first + " (" + std::to_string(instances) + ")";
					std::string accur = "NA";
					
					// Make sure there was at least one prediction to report accuracy
					if (!std::isnan(accuracy)) {
						std::stringstream accuracyStream;
						accuracyStream << std::setprecision(2) << std::fixed << accuracy << "%";
						accur = accuracyStream.str();
					}
					stream <<
						std::setw(7)  << "STATS"                    << " " <<
						std::setw(12) << "MONITORING"               << " " <<
						std::setw(26) << "TASK-TYPE (INSTANCES)"    << " " <<
						std::setw(20) << typeLabel                  << "\n";
					stream <<
						std::setw(7)  << "STATS"                    << " "   <<
						std::setw(12) << "MONITORING"               << " "   <<
						std::setw(26) << "UNITARY COST AVG / STDEV" << " "   <<
						std::setw(10) << avgCost                    << " / " <<
						std::setw(10) << stdevCost                  << "\n";
					stream <<
						std::setw(7)  << "STATS"                    << " " <<
						std::setw(12) << "MONITORING"               << " " <<
						std::setw(26) << "PREDICTION ACCURACY (%)"  << " " <<
						std::setw(10) << accur                      << "\n";
					stream << "+-----------------------------+\n";
				}
			}
			stream << "\n";
		}
	}
	
	//! \brief Initialize a task's monitoring statistics
	//! \param parentStatistics The parent task's statistics
	//! \param taskStatistics The task's statistics
	//! \param parentPredictions The parent task's predictions
	//! \param taskPredictions The task's predictions
	//! \param label The tasktype
	//! \param cost The task's computational cost
	static void taskCreated(
		TaskStatistics  *parentStatistics,
		TaskStatistics  *taskStatistics,
		TaskPredictions *parentPredictions,
		TaskPredictions *taskPredictions,
		const std::string &label,
		size_t cost
	);
	
	//! \brief Predict the execution time of a task
	//! \param taskPredictions The predictions of the task
	//! \param label The tasktype
	//! \param cost The task's computational task
	static void predictTime(TaskPredictions *taskPredictions, const std::string &label, size_t cost);
	
	//! \brief Start time monitoring for a task
	//! \param taskStatistics The task's statistics
	//! \param execStatus The timing status to start
	//! \return The status before the change
	static monitoring_task_status_t startTiming(TaskStatistics *taskStatistics, monitoring_task_status_t execStatus);
	
	//! \brief Stop time monitoring for a task
	//! \param[in,out] taskStatistics The task's statistics
	//! \param[in,out] taskPredictions The predictions of the task
	//! \param[out] ancestorsUpdated The number of ancestors that this task has
	//! updated during shutdown of timing monitoring
	//! \return The status before the change
	static monitoring_task_status_t stopTiming(TaskStatistics *taskStatistics, TaskPredictions *taskPredictions, int &ancestorsUpdated);
	
	//! \brief Get the average unitary time value of a tasktype (normalized using cost)
	//! \param label The tasktype
	static double getAverageTimePerUnitOfCost(const std::string &label);
	
	//! \brief Insert an unitary time value (normalized using cost) into the 
	//! appropriate TasktypePredictions structure
	//! \param label The tasktype
	//! \param unitaryTime The time per unit of cost to insert
	static void insertTimePerUnitOfCost(const std::string &label, double unitaryTime);
	
	//! \brief Get the average unitary time values of all the tasktypes
	//! being monitored
	//! \param[out] labels The reference of a vector in which all the available
	//! tasktypes will be inserted
	//! \param[out] unitaryTimes The reference of a vector in which the
	//! times per unit of cost will be inserted
	static void getAverageTimesPerUnitOfCost(
		std::vector<std::string> &labels,
		std::vector<double> &unitaryTimes
	);
	
};

#endif // TASK_MONITOR_HPP
