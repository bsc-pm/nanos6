/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_MONITOR_HPP
#define TASK_MONITOR_HPP

#include <sys/time.h>

#include "MonitoringSupport.hpp"
#include "tasks/Task.hpp"


class TaskMonitor {

private:

	//! Timestamp of the beginning of the execution
	timeval _initialTimestamp;

public:

	inline TaskMonitor()
	{
		gettimeofday(&_initialTimestamp, nullptr);
	}

	//! \brief Initialize a task's monitoring statistics
	//!
	//! \param[in,out] task The task
	//! \param[in,out] parent The task's parent
	void taskCreated(Task *task, Task *parent) const;

	//! \brief Re-initialize a task's monitoring statistics
	//!
	//! \param[out] taks The task
	void taskReinitialized(Task *task) const;

	//! \brief Start time monitoring for a task
	//!
	//! \param[in,out] task The task
	//! \param[in] execStatus The timing status to start
	void taskStarted(Task *task, monitoring_task_status_t execStatus) const;

	//! \brief Accumulate statistics when the task completes user code
	//!
	//! \param[in,out] task The task
	void taskCompletedUserCode(Task *task) const;

	//! \brief Stop time monitoring for a task
	//!
	//! \param[in,out] task The task
	void taskFinished(Task *task) const;

	//! \brief Display task statistics
	//!
	//! \param[out] stream The output stream
	void displayStatistics(std::stringstream &stream) const;

};

#endif // TASK_MONITOR_HPP
