/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <config.h>
#include <fstream>

#include "Monitoring.hpp"


ConfigVariable<bool> Monitoring::_enabled("monitoring.enabled", true);
ConfigVariable<bool> Monitoring::_verbose("monitoring.verbose", true);
ConfigVariable<bool> Monitoring::_wisdomEnabled("monitoring.wisdom", false);
ConfigVariable<std::string> Monitoring::_outputFile("monitoring.verbose_file", "output-monitoring.txt");
JsonFile *Monitoring::_wisdom(nullptr);
TaskMonitor *Monitoring::_taskMonitor(nullptr);
CPUMonitor *Monitoring::_cpuMonitor(nullptr);
CPUUsagePredictor *Monitoring::_cpuUsagePredictor(nullptr);
WorkloadPredictor *Monitoring::_workloadPredictor(nullptr);


//    MONITORING    //

void Monitoring::initialize()
{
	if (_enabled) {
#if CHRONO_ARCH
		// Start measuring time to compute the tick conversion rate
		TickConversionUpdater::initialize();
#endif
		// Create all the monitors and predictors
		_taskMonitor = new TaskMonitor();
		_cpuMonitor = new CPUMonitor();
		_cpuUsagePredictor = new CPUUsagePredictor();
		_workloadPredictor = new WorkloadPredictor();
		assert(_taskMonitor != nullptr);
		assert(_cpuMonitor != nullptr);
		assert(_cpuUsagePredictor != nullptr);
		assert(_workloadPredictor != nullptr);

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

		if (_verbose) {
			displayStatistics();
		}

#if CHRONO_ARCH
		// Destroy the tick conversion updater service
		TickConversionUpdater::shutdown();
#endif

		// Delete all predictors and monitors
		delete _workloadPredictor;
		delete _cpuMonitor;
		delete _cpuUsagePredictor;
		delete _taskMonitor;
		_workloadPredictor = nullptr;
		_cpuMonitor = nullptr;
		_cpuUsagePredictor = nullptr;
		_taskMonitor = nullptr;

		_enabled.setValue(false);
	}
}

//    TASKS    //

void Monitoring::taskCreated(Task *task)
{
	assert(task != nullptr);

	// Create task statistics
	if (_enabled) {
		TaskStatistics *taskStatistics = task->getTaskStatistics();
		assert(taskStatistics != nullptr);

		// Construct the object with the reserved space
		new (taskStatistics) TaskStatistics();
	}

	if (_enabled && !task->isTaskfor()) {
		// Retrieve information about the task
		TaskStatistics *parentStatistics = (task->getParent() != nullptr ? task->getParent()->getTaskStatistics() : nullptr);
		TaskStatistics *taskStatistics = task->getTaskStatistics();
		const std::string &label = task->getLabel();
		size_t cost = (task->hasCost() ? task->getCost() : DEFAULT_COST);

		// Populate task statistic structures and predict metrics
		_taskMonitor->taskCreated(parentStatistics, taskStatistics, label, cost);
		_taskMonitor->predictTime(taskStatistics, label, cost);

		// Account this task in workloads
		_workloadPredictor->taskCreated(taskStatistics);
	}
}

void Monitoring::taskChangedStatus(Task *task, monitoring_task_status_t newStatus)
{
	assert(task != nullptr);

	if (_enabled && !task->isTaskfor()) {
		// Start timing for the appropriate stopwatch
		const monitoring_task_status_t oldStatus = _taskMonitor->startTiming(task->getTaskStatistics(), newStatus);

		// Update workload statistics only after a change of status
		if (oldStatus != newStatus) {
			// Account this task in the appropriate workload
			_workloadPredictor->taskChangedStatus(task->getTaskStatistics(), oldStatus, newStatus);
		}
	}
}

void Monitoring::taskCompletedUserCode(Task *task)
{
	assert(task != nullptr);

	if (_enabled && !task->isTaskfor()) {
		// Account the task's elapsed execution time in predictions
		_workloadPredictor->taskCompletedUserCode(task->getTaskStatistics());
	}
}

void Monitoring::taskFinished(Task *task)
{
	assert(task != nullptr);

	if (_enabled && !task->isTaskfor()) {
		// Number of ancestors updated by this task in TaskMonitor
		int ancestorsUpdated = 0;

		// Mark task as completely executed
		const monitoring_task_status_t oldStatus = _taskMonitor->stopTiming(task->getTaskStatistics(), ancestorsUpdated);

		// Account this task in workloads
		_workloadPredictor->taskFinished(task->getTaskStatistics(), oldStatus, ancestorsUpdated);
	}
}


//    CPUS    //

void Monitoring::cpuBecomesIdle(int cpuId)
{
	if (_enabled) {
		_cpuMonitor->cpuBecomesIdle(cpuId);
	}
}

void Monitoring::cpuBecomesActive(int cpuId)
{
	if (_enabled) {
		_cpuMonitor->cpuBecomesActive(cpuId);
	}
}


//    PREDICTORS    //

double Monitoring::getPredictedElapsedTime()
{
	if (_enabled) {
		const double cpuUtilization = _cpuMonitor->getTotalActiveness();
		const double instantiated = _workloadPredictor->getPredictedWorkload(instantiated_load);
		const double finished = _workloadPredictor->getPredictedWorkload(finished_load);

		// Convert completion times -- current elapsed execution time of tasks
		// that have not finished execution yet -- from ticks to microseconds
		Chrono completionTime(_workloadPredictor->getTaskCompletionTimes());
		const double completion = ((double) completionTime);

		double timeLeft = ((instantiated - finished - completion) / cpuUtilization);

		// Check if the elapsed time substracted from the predictions underflows
		return (timeLeft < 0.0 ? 0.0 : timeLeft);
	}

	return 0.0;
}

//    PRIVATE METHODS    //

void Monitoring::displayStatistics()
{
	// Try opening the output file
	std::ios_base::openmode openMode = std::ios::out;
	std::ofstream output(_outputFile.getValue(), openMode);
	FatalErrorHandler::warnIf(
		!output.is_open(),
		"Could not create or open the verbose file: ", _outputFile.getValue(), ". Using standard output."
	);

	// Retrieve statistics from every monitor and predictor
	std::stringstream outputStream;
	_taskMonitor->displayStatistics(outputStream);
	_cpuMonitor->displayStatistics(outputStream);
	_cpuUsagePredictor->displayStatistics(outputStream);
	_workloadPredictor->displayStatistics(outputStream);

	if (output.is_open()) {
		output << outputStream.str();
		output.close();
	} else {
		std::cout << outputStream.str();
	}
}

void Monitoring::loadMonitoringWisdom()
{
	// Create a representation of the system file as a JsonFile
	_wisdom = new JsonFile("./.nanos6_monitoring_wisdom.json");
	assert(_wisdom != nullptr);

	// Try to populate the JsonFile with the system file's data
	_wisdom->loadData();

	// Navigate through the file and extract the unitary time of each tasktype
	_wisdom->getRootNode()->traverseChildrenNodes(
		[&](const std::string &label, const JsonNode<> &metricsNode) {
			// For each tasktype, check if the unitary time is available
			if (metricsNode.dataExists("unitary_time")) {
				// Insert the metric data for this tasktype into accumulators
				bool converted = false;
				double metricValue = metricsNode.getData("unitary_time", converted);
				if (converted) {
					_taskMonitor->insertTimePerUnitOfCost(label, metricValue);
				}
			}
		}
	);
}

void Monitoring::storeMonitoringWisdom()
{
	// Gather monitoring data for all tasktypes
	std::vector<std::string> labels;
	std::vector<double> unitaryTimes;
	_taskMonitor->getAverageTimesPerUnitOfCost(labels, unitaryTimes);

	assert(_wisdom != nullptr);

	// The file's root node
	JsonNode<> *rootNode = _wisdom->getRootNode();
	for (size_t i = 0; i < labels.size(); ++i) {
		// Avoid storing information about the main task
		if (labels[i] != "main") {
			// A node for metrics (currently only unitary time)
			JsonNode<double> taskTypeValuesNode;
			taskTypeValuesNode.addData("unitary_time", unitaryTimes[i]);

			// Add the metrics to the root node of the file
			rootNode->addChildNode(labels[i], taskTypeValuesNode);
		}
	}

	// Store the data from the JsonFile in the system file
	_wisdom->storeData();

	// Delete the file as it is no longer needed
	delete _wisdom;
}
