/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef MONITORING_HPP
#define MONITORING_HPP

#include "CPUMonitor.hpp"
#include "CPUUsagePredictor.hpp"
#include "TaskMonitor.hpp"
#include "WorkloadPredictor.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "support/JsonFile.hpp"
#include "support/config/ConfigVariable.hpp"
#include "tasks/Task.hpp"


class Monitoring {

private:

	//! Whether monitoring has to be performed or not
	static ConfigVariable<bool> _enabled;

	//! Whether verbose mode is enabled
	static ConfigVariable<bool> _verbose;

	//! Whether the wisdom mechanism is enabled
	static ConfigVariable<bool> _wisdomEnabled;

	//! The file where output must be saved when verbose mode is enabled
	static ConfigVariable<std::string> _outputFile;

	//! A Json file for monitoring data
	static JsonFile *_wisdom;

	//! A monitor that handles task statistics
	static TaskMonitor *_taskMonitor;

	//! A monitor that handles CPU statistics
	static CPUMonitor *_cpuMonitor;

	//! A predictor that deals with computing cpu utilization estimates
	static CPUUsagePredictor *_cpuUsagePredictor;

	//! A predictor that deals with runtime workload predictions
	static WorkloadPredictor *_workloadPredictor;

private:

	//! \brief Display monitoring statistics
	static void displayStatistics();

	//! \brief Try to load previous monitoring data into accumulators
	static void loadMonitoringWisdom();

	//! \brief Store monitoring data for future executions as warmup data
	static void storeMonitoringWisdom();

public:

	//    MONITORING    //

	//! \brief Initialize monitoring
	static void initialize();

	//! \brief Shutdown monitoring
	static void shutdown();

	//! \brief Check whether monitoring is enabled
	static inline bool isEnabled()
	{
		return _enabled;
	}

	//    TASKS    //

	//! \brief Gather basic information about a task when it is created
	//!
	//! \param[in,out] task The task to gather information about
	static void taskCreated(Task *task);

	//! \brief Propagate monitoring operations after a task has changed its
	//! execution status
	//!
	//! \param[in,out] task The task that's changing status
	//! \param[in] newStatus The new execution status of the task
	static void taskChangedStatus(Task *task, monitoring_task_status_t newStatus);

	//! \brief Propagate monitoring operations after a task has
	//! completed user code execution
	//!
	//! \param[in,out] task The task that has completed the execution
	static void taskCompletedUserCode(Task *task);

	//! \brief Propagate monitoring operations after a task has finished
	//!
	//! \param[in,out] task The task that has finished
	static void taskFinished(Task *task);

	//! \brief Get the size needed to create a TaskStatistics object
	//!
	//! \return TaskStatistics size or 0 if Monitoring is disabled
	static inline size_t getTaskStatisticsSize()
	{
		if (_enabled) {
			return sizeof(TaskStatistics);
		}

		return 0;
	}

	//    CPUS    //

	//! \brief Propagate monitoring operations when a CPU becomes idle
	//!
	//! \param[in] cpuId The identifier of the CPU
	static void cpuBecomesIdle(int cpuId);

	//! \brief Propagate monitoring operations when a CPU becomes active
	//!
	//! \param[in] cpuId The identifier of the CPU
	static void cpuBecomesActive(int cpuId);

	//    PREDICTORS    //

	//! \brief Poll the expected time until completion of the current execution
	//!
	//! \return An estimation of the time to completion in microseconds
	static double getPredictedElapsedTime();

};

#endif // MONITORING_HPP
