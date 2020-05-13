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
CPUMonitor *Monitoring::_cpuMonitor(nullptr);
TaskMonitor *Monitoring::_taskMonitor(nullptr);
Monitoring::accumulator_t Monitoring::_cpuUsageAccuracyAccum;
double Monitoring::_cpuUsagePrediction;
bool Monitoring::_cpuUsageAvailable(false);


//    MONITORING    //

void Monitoring::preinitialize()
{
	if (_enabled) {
#if CHRONO_ARCH
		// Start measuring time to compute the tick conversion rate
		TickConversionUpdater::initialize();
#endif
		// Create all the monitors and predictors
		_taskMonitor = new TaskMonitor();
		assert(_taskMonitor != nullptr);

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

void Monitoring::initialize()
{
	if (_enabled) {
		// Create all the monitors and predictors
		_cpuMonitor = new CPUMonitor();
		assert(_cpuMonitor != nullptr);
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
		assert(_cpuMonitor != nullptr);
		assert(_taskMonitor != nullptr);

		delete _cpuMonitor;
		delete _taskMonitor;
		_cpuMonitor = nullptr;
		_taskMonitor = nullptr;
		_enabled.setValue(false);
	}
}


//    TASKS    //

void Monitoring::taskCreated(Task *task)
{
	if (_enabled) {
		assert(task != nullptr);
		assert(_taskMonitor != nullptr);

		TaskStatistics *taskStatistics = task->getTaskStatistics();
		assert(taskStatistics != nullptr);

		// Construct the object with the reserved space
		new (taskStatistics) TaskStatistics();

		// Only take the task into account for predictions if it is a basic
		// task or an original Taskfor, never a collaborator
		if (!task->isTaskfor() || (task->isTaskfor() && !task->isRunnable())) {
			// Populate task statistic structures and predict metrics
			Task *parent = task->getParent();
			_taskMonitor->taskCreated(task, parent);
		}
	}
}

void Monitoring::taskReinitialized(Task *task)
{
	if (_enabled) {
		// Make sure this is a Taskfor
		assert(task != nullptr);
		assert(task->isTaskfor());
		assert(_taskMonitor != nullptr);

		// Reset task statistics
		_taskMonitor->taskReinitialized(task);
	}
}

void Monitoring::taskChangedStatus(Task *task, monitoring_task_status_t newStatus)
{
	if (_enabled) {
		assert(_taskMonitor != nullptr);

		// Start timing for the appropriate stopwatch
		_taskMonitor->startTiming(task, newStatus);
	}
}

void Monitoring::taskCompletedUserCode(Task *task)
{
	if (_enabled) {
		assert(_taskMonitor != nullptr);

		// Account the task's elapsed execution for predictions
		_taskMonitor->taskCompletedUserCode(task);
	}
}

void Monitoring::taskFinished(Task *task)
{
	if (_enabled) {
		assert(task != nullptr);
		assert(_taskMonitor != nullptr);

		// If the task is a taskfor collaborator, aggregate statistics in the
		// parent (source taskfor). Otherwise normal task behavior
		if (task->isTaskfor() && task->isRunnable()) {
			Task *source = task->getParent();
			assert(source != nullptr);
			assert(source->isTaskfor());

			_taskMonitor->taskforCollaboratorFinished(task, source);
		} else {
			// Mark task as completely executed
			_taskMonitor->stopTiming(task);
		}
	}
}


//    CPUS    //

void Monitoring::cpuBecomesIdle(int cpuId)
{
	if (_enabled) {
		assert(_cpuMonitor != nullptr);

		_cpuMonitor->cpuBecomesIdle(cpuId);
	}
}

void Monitoring::cpuBecomesActive(int cpuId)
{
	if (_enabled) {
		assert(_cpuMonitor != nullptr);

		_cpuMonitor->cpuBecomesActive(cpuId);
	}
}


//    PREDICTORS    //

double Monitoring::getPredictedWorkload(MonitoringWorkloads::workload_t loadId)
{
	// FIXME TODO
	/*
	if (_enabled) {
		assert(_taskMonitor != nullptr);
		assert(_workloadMonitor != nullptr);

		workloads_map_t workloadsMap = _workloadMonitor->getWorkloadsMapReference();

		double totalTime = 0.0;
		for (auto const &it : workloadsMap) {
			assert(it.second != nullptr);

			totalTime += (
				it.second->getAccumulatedCost(loadId) *
				_taskMonitor->getAverageTimePerUnitOfCost(it.first)
			);
		}

		return totalTime;
	}

	return 0.0;
	*/
}

double Monitoring::getCPUUsagePrediction(size_t time)
{
	// FIXME TODO
	/*
	if (_enabled) {
		assert(_cpuMonitor != nullptr);
		assert(_taskMonitor != nullptr);
		assert(_workloadMonitor != nullptr);

		// Retrieve the current ready workload
		double readyLoad      = getPredictedWorkload(ready_load);
		double executingLoad  = getPredictedWorkload(executing_load);
		double readyTasks     = (double) _workloadMonitor->getNumInstances(ready_load);
		double executingTasks = (double) _workloadMonitor->getNumInstances(executing_load);
		double numberCPUs     = (double) _cpuMonitor->getNumCPUs();

		// To account accuracy, make sure this isn't the first prediction
		if (!_cpuUsageAvailable) {
			_cpuUsageAvailable = true;
		} else {
			// Compute the accuracy of the last prediction
			double utilization = _cpuMonitor->getTotalActiveness();
			double error = (std::abs(utilization - _cpuUsagePrediction)) / numberCPUs;
			double accuracy = 100.0 - (100.0 * error);
			_cpuUsageAccuracyAccum(accuracy);
		}

		// Make sure that the prediction is:
		// - At most the amount of tasks or the amount of time available
		// - At least 1 CPU for runtime-related operations
		_cpuUsagePrediction = std::min(executingTasks + readyTasks, (readyLoad + executingLoad) / (double) time);
		_cpuUsagePrediction = std::max(_cpuUsagePrediction, 1.0);

		return _cpuUsagePrediction;
	}
	*/
	return 0.0;
}

double Monitoring::getPredictedElapsedTime()
{
	/*
	if (_enabled) {
		assert(_cpuMonitor != nullptr);
		assert(_workloadMonitor != nullptr);

		const double cpuUtilization = _cpuMonitor->getTotalActiveness();
		const double instantiated = getPredictedWorkload(instantiated_load);
		const double finished = getPredictedWorkload(finished_load);

		// Convert completion times -- current elapsed execution time of tasks
		// that have not finished execution yet -- from ticks to microseconds
		Chrono completionTime(_workloadMonitor->getTaskCompletionTimes());
		const double completion = ((double) completionTime);

		double timeLeft = ((instantiated - finished - completion) / cpuUtilization);

		// Check if the elapsed time substracted from the predictions underflows
		return (timeLeft < 0.0 ? 0.0 : timeLeft);
	}
	*/
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

	// Retrieve statistics from every monitor
	std::stringstream outputStream;
	_taskMonitor->displayStatistics(outputStream);
	_cpuMonitor->displayStatistics(outputStream);

	// Print the statistics of every prediction heuristic
	outputStream << std::left << std::fixed << std::setprecision(2) << "\n";
	outputStream << "+-----------------------------+\n";
	outputStream << "|    CPU Usage Predictions    |\n";
	outputStream << "+-----------------------------+\n";
	if (_cpuUsageAvailable) {
		outputStream << "  MEAN ACCURACY: " << BoostAcc::mean(_cpuUsageAccuracyAccum) << "%\n";
	} else {
		outputStream << "  MEAN ACCURACY: NA\n";
	}
	outputStream << "+-----------------------------+\n\n";

	if (output.is_open()) {
		output << outputStream.str();
		output.close();
	} else {
		std::cout << outputStream.str();
	}
}

void Monitoring::loadMonitoringWisdom()
{
	// FIXME TODO
	/*
	assert(_taskMonitor != nullptr);

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
	*/
}

void Monitoring::storeMonitoringWisdom()
{
	// FIXME TODO
	/*
	assert(_taskMonitor != nullptr);

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
	*/
}
