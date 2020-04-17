/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <config.h>
#include <fstream>

#include "CPUUsagePredictor.hpp"
#include "Monitoring.hpp"


ConfigVariable<bool> Monitoring::_enabled("monitoring.enabled", true);
ConfigVariable<bool> Monitoring::_verbose("monitoring.verbose", true);
ConfigVariable<bool> Monitoring::_wisdomEnabled("monitoring.wisdom", false);
ConfigVariable<std::string> Monitoring::_outputFile("monitoring.verbose_file", "output-monitoring.txt");
JsonFile *Monitoring::_wisdom(nullptr);


//    MONITORING    //

void Monitoring::initialize()
{
	if (_enabled) {
		#if CHRONO_ARCH
			// Start measuring time to compute the tick conversion rate
			TickConversionUpdater::initialize();
		#endif

		// Initialize the task monitoring module
		TaskMonitor::initialize();

		// Initialize the CPU monitoring module
		CPUMonitor::initialize();

		// Initialize the workload predictor
		WorkloadPredictor::initialize();

		#if CHRONO_ARCH
			// Stop measuring time and compute the tick conversion rate
			TickConversionUpdater::finishUpdate();
		#endif

		if (_wisdomEnabled) {
			// Try to load data from previous executions
			loadMonitoringWisdom();
		}
	}
}

void Monitoring::shutdown()
{
	if (_enabled) {
		if (_wisdomEnabled) {
			// Store monitoring data for future executions
			storeMonitoringWisdom();
		}

		#if CHRONO_ARCH
			// Destroy the tick conversion updater service
			TickConversionUpdater::shutdown();
		#endif

		// Display monitoring statistics
		displayStatistics();

		// Propagate shutdown to the workload predictor
		WorkloadPredictor::shutdown();

		// Propagate shutdown to the CPU monitoring module
		CPUMonitor::shutdown();

		// Propagate shutdown to the task monitoring module
		TaskMonitor::shutdown();

		_enabled.setValue(false);
	}
}

void Monitoring::displayStatistics()
{
	if (_enabled && _verbose) {
		// Try opening the output file
		std::ios_base::openmode openMode = std::ios::out;
		std::ofstream output(_outputFile.getValue(), openMode);
		FatalErrorHandler::warnIf(
			!output.is_open(),
			"Could not create or open the verbose file: ",
			_outputFile.getValue(),
			". Using standard output."
		);

		// Retrieve statistics from every module / predictor
		std::stringstream outputStream;
		CPUMonitor::displayStatistics(outputStream);
		CPUUsagePredictor::displayStatistics(outputStream);
		TaskMonitor::displayStatistics(outputStream);
		WorkloadPredictor::displayStatistics(outputStream);

		if (output.is_open()) {
			// Output into the file and close it
			output << outputStream.str();
			output.close();
		} else {
			std::cout << outputStream.str();
		}
	}
}

bool Monitoring::isEnabled()
{
	return _enabled;
}


//    TASKS    //

void Monitoring::taskCreated(Task *task)
{
	assert(task != nullptr);
	if (_enabled && !task->isTaskfor()) {
		// Retrieve information about the task
		TaskStatistics  *parentStatistics  = (task->getParent() != nullptr ? task->getParent()->getTaskStatistics() : nullptr);
		TaskStatistics  *taskStatistics    = task->getTaskStatistics();
		const std::string &label = task->getLabel();
		size_t cost = (task->hasCost() ? task->getCost() : DEFAULT_COST);

		// Create task statistic structures and predict its execution time
		TaskMonitor::taskCreated(parentStatistics, taskStatistics, label, cost);
		TaskMonitor::predictTime(taskStatistics, label, cost);

		// Account this task in workloads
		WorkloadPredictor::taskCreated(taskStatistics);
	}
}

void Monitoring::taskChangedStatus(Task *task, monitoring_task_status_t newStatus)
{
	assert(task != nullptr);
	if (_enabled && !task->isTaskfor()) {
		// Start timing for the appropriate stopwatch
		const monitoring_task_status_t oldStatus = TaskMonitor::startTiming(task->getTaskStatistics(), newStatus);

		// Update workload statistics only after a change of status
		if (oldStatus != newStatus) {
			// Account this task in the appropriate workload
			WorkloadPredictor::taskChangedStatus(task->getTaskStatistics(), oldStatus, newStatus);
		}
	}
}

void Monitoring::taskCompletedUserCode(Task *task)
{
	assert(task != nullptr);
	if (_enabled && !task->isTaskfor()) {
		// Account the task's elapsed execution time in predictions
		WorkloadPredictor::taskCompletedUserCode(task->getTaskStatistics());
	}
}

void Monitoring::taskFinished(Task *task)
{
	assert(task != nullptr);
	if (_enabled && !task->isTaskfor()) {
		// Number of ancestors updated by this task in TaskMonitor
		int ancestorsUpdated = 0;

		// Mark task as completely executed
		const monitoring_task_status_t oldStatus = TaskMonitor::stopTiming(task->getTaskStatistics(), ancestorsUpdated);

		// Account this task in workloads
		WorkloadPredictor::taskFinished(task->getTaskStatistics(), oldStatus, ancestorsUpdated);
	}
}


//    THREADS    //

void Monitoring::initializeThread()
{
	// Empty thread API
}

void Monitoring::shutdownThread()
{
	// Empty thread API
}


//    CPUS    //

void Monitoring::cpuBecomesIdle(int cpuId)
{
	if (_enabled) {
		CPUMonitor::cpuBecomesIdle(cpuId);
	}
}

void Monitoring::cpuBecomesActive(int cpuId)
{
	if (_enabled) {
		CPUMonitor::cpuBecomesActive(cpuId);
	}
}


//    PREDICTORS    //

double Monitoring::getPredictedElapsedTime()
{
	if (_enabled) {
		const double cpuUtilization = CPUMonitor::getTotalActiveness();
		const double instantiated   = WorkloadPredictor::getPredictedWorkload(instantiated_load);
		const double finished       = WorkloadPredictor::getPredictedWorkload(finished_load);

		// Convert completion times -- current elapsed execution time of tasks
		// that have not finished execution yet -- from ticks to microseconds
		Chrono completionTime(WorkloadPredictor::getTaskCompletionTimes());
		const double completion = ((double) completionTime);

		double timeLeft = ((instantiated - finished - completion) / cpuUtilization);

		// Check if the elapsed time substracted from the predictions underflows
		return (timeLeft < 0.0 ? 0.0 : timeLeft);
	}

	return 0.0;
}
