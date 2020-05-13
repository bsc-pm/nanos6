/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_MONITOR_HPP
#define TASK_MONITOR_HPP

#include "TaskStatistics.hpp"
#include "TasktypeStatistics.hpp"
#include "tasks/Task.hpp"


class TaskMonitor {

private:

	//! \brief Maps task status identifiers to workload identifiers
	//!
	//! \param[in] taskStatus The task status
	//!
	//! \return The workload id related to the task status
	inline MonitoringWorkloads::workload_t getLoadId(monitoring_task_status_t taskStatus) const
	{
		switch (taskStatus) {
			case executing_status:
				return MonitoringWorkloads::executing_load;
			case ready_status:
				return MonitoringWorkloads::ready_load;
			case pending_status:
			case blocked_status:
			case runtime_status:
				return MonitoringWorkloads::null_workload;
			default:
				return MonitoringWorkloads::null_workload;
		}
	}


public:

	//! \brief Display task statistics
	//!
	//! \param[out] stream The output stream
	void displayStatistics(std::stringstream &stream) const;

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
	void startTiming(Task *task, monitoring_task_status_t execStatus) const;

	//! \brief Accumulate statistics when the task completes user code
	//!
	//! \param[in,out] task The task
	void taskCompletedUserCode(Task *task) const;

	//! \brief Stop time monitoring for a task
	//!
	//! \param[in,out] task The task
	void stopTiming(Task *task) const;

	//! \brief Aggregate a collaborator's statistics into the parent Taskfor
	//!
	//! \param[in,out] task The collaborator
	//! \param[in,out] source The parent Taskfor
	void taskforCollaboratorFinished(Task *task, Task *source) const;

};

#endif // TASK_MONITOR_HPP
