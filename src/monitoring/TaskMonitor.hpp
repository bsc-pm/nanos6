/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
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

	typedef std::map<std::string, TasktypePredictions *> tasktype_map_t;

	//! Maps TasktypePredictions by task labels
	tasktype_map_t _tasktypeMap;

	//! Spinlock that ensures atomic access within the tasktype map
	SpinLock _spinlock;

public:

	inline TaskMonitor() :
		_tasktypeMap(),
		_spinlock()
	{
	}

	inline ~TaskMonitor()
	{
		// Destroy all the task type statistics
		for (auto &it : _tasktypeMap) {
			if (it.second != nullptr) {
				delete it.second;
			}
		}
	}

	//! \brief Display task statistics
	//!
	//! \param[out] stream The output stream
	void displayStatistics(std::stringstream &stream);

	//! \brief Initialize a task's monitoring statistics
	//!
	//! \param[in,out] parentStatistics The parent task's statistics
	//! \param[in,out] taskStatistics The task's statistics
	//! \param[in] label The tasktype
	//! \param[in] cost The task's computational cost
	void taskCreated(
		TaskStatistics  *parentStatistics,
		TaskStatistics  *taskStatistics,
		const std::string &label,
		size_t cost
	);

	//! \brief Predict the execution time of a task
	//!
	//! \param[in,out] taskStatistics The statistics of the task
	//! \param[in] label The tasktype
	//! \param[in] cost The task's computational task
	void predictTime(TaskStatistics *taskStatistics, const std::string &label, size_t cost);

	//! \brief Start time monitoring for a task
	//!
	//! \param[in,out] taskStatistics The task's statistics
	//! \param[in] execStatus The timing status to start
	//!
	//! \return The status before the change
	monitoring_task_status_t startTiming(TaskStatistics *taskStatistics, monitoring_task_status_t execStatus);

	//! \brief Stop time monitoring for a task
	//!
	//! \param[in,out] taskStatistics The task's statistics
	//! \param[out] ancestorsUpdated The number of ancestors that this task has
	//! updated during shutdown of timing monitoring
	//!
	//! \return The status before the change
	monitoring_task_status_t stopTiming(TaskStatistics *taskStatistics, int &ancestorsUpdated);

	//! \brief Get the average unitary time value of a tasktype (normalized using cost)
	//!
	//! \param[in] label The tasktype
	double getAverageTimePerUnitOfCost(const std::string &label);

	//! \brief Insert an unitary time value (normalized using cost) into the
	//! appropriate TasktypePredictions structure
	//!
	//! \param[in] label The tasktype
	//! \param[in] unitaryTime The time per unit of cost to insert
	void insertTimePerUnitOfCost(const std::string &label, double unitaryTime);

	//! \brief Get the average unitary time values of all the tasktypes
	//! being monitored
	//!
	//! \param[out] labels The reference of a vector in which all the available
	//! tasktypes will be inserted
	//! \param[out] unitaryTimes The reference of a vector in which the
	//! times per unit of cost will be inserted
	void getAverageTimesPerUnitOfCost(
		std::vector<std::string> &labels,
		std::vector<double> &unitaryTimes
	);

};

#endif // TASK_MONITOR_HPP
