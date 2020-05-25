/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <config.h>
#include <fstream>

#include "Monitoring.hpp"
#include "TasktypeStatistics.hpp"
#include "tasks/TaskInfo.hpp"


ConfigVariable<bool> Monitoring::_enabled("monitoring.enabled", true);
ConfigVariable<bool> Monitoring::_verbose("monitoring.verbose", true);
ConfigVariable<bool> Monitoring::_wisdomEnabled("monitoring.wisdom", false);
ConfigVariable<std::string> Monitoring::_outputFile("monitoring.verbose_file", "output-monitoring.txt");
JsonFile *Monitoring::_wisdom(nullptr);
CPUMonitor *Monitoring::_cpuMonitor(nullptr);
TaskMonitor *Monitoring::_taskMonitor(nullptr);
double Monitoring::_predictedCPUUsage;
bool Monitoring::_predictedCPUUsageAvailable(false);


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
		_taskMonitor->taskStarted(task, newStatus);
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

		// Mark task as completely executed
		_taskMonitor->taskFinished(task);
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

double Monitoring::getPredictedCPUUsage(size_t time)
{
	if (_enabled) {
		assert(_cpuMonitor != nullptr);

		if (!_predictedCPUUsageAvailable) {
			_predictedCPUUsageAvailable = true;
		}

		double currentWorkload = 0.0;
		size_t currentActiveInstances = 0;
		size_t currentPredictionlessInstances = 0;
		TaskInfo::processAllTasktypes(
			[&](const std::string &, TasktypeData &tasktypeData) {
				TasktypeStatistics &statistics = tasktypeData.getTasktypeStatistics();
				Chrono completedChrono(statistics.getCompletedTime());
				double completedTime = ((double) completedChrono);
				size_t accumulatedCost = statistics.getAccumulatedCost();
				double accumulatedTime = statistics.getTimePrediction(accumulatedCost);

				if (accumulatedTime > completedTime) {
					currentWorkload += (accumulatedTime - completedTime);
					currentActiveInstances += statistics.getNumInstances();
					currentPredictionlessInstances += statistics.getNumPredictionlessInstances();
				}
			}
		);

		// At least one CPU
		double predictedUsage = 1.0;

		// If there are any, at least the number of predictionless instances
		if (currentPredictionlessInstances > 0) {
			predictedUsage = currentPredictionlessInstances;
		}

		// Add the minimum between the number of tasks with prediction, and
		// the current workload in time divided by the required time
		predictedUsage += std::min(
			(double) currentActiveInstances,
			(currentWorkload / (double) time)
		);
		_predictedCPUUsage = predictedUsage;

		return predictedUsage;
	}

	return 0.0;
}

double Monitoring::getPredictedElapsedTime()
{
	if (_enabled) {
		assert(_cpuMonitor != nullptr);

		double currentWorkload = 0.0;
		TaskInfo::processAllTasktypes(
			[&](const std::string &, TasktypeData &tasktypeData) {
				TasktypeStatistics &statistics = tasktypeData.getTasktypeStatistics();
				Chrono completedChrono(statistics.getCompletedTime());
				double completedTime = ((double) completedChrono);
				size_t accumulatedCost = statistics.getAccumulatedCost();
				double accumulatedTime = statistics.getTimePrediction(accumulatedCost);

				if (accumulatedTime > completedTime) {
					currentWorkload += (accumulatedTime - completedTime);
				}
			}
		);

		// Check how active CPUs currently are
		double currentCPUActiveness = _cpuMonitor->getTotalActiveness();

		// Check if the elapsed time substracted from the predictions underflows
		return (currentWorkload < 0.0 ? 0.0 : (currentWorkload / currentCPUActiveness));
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

	// Retrieve statistics from every monitor
	std::stringstream outputStream;
	_taskMonitor->displayStatistics(outputStream);
	_cpuMonitor->displayStatistics(outputStream);

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
