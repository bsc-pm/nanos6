/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "MonitoringSupport.hpp"
#include "TaskMonitor.hpp"
#include "TasktypeStatistics.hpp"
#include "TaskStatistics.hpp"
#include "hardware-counters/HardwareCounters.hpp"
#include "hardware-counters/SupportedHardwareCounters.hpp"
#include "hardware-counters/TaskHardwareCounters.hpp"
#include "tasks/Task.hpp"
#include "tasks/Taskfor.hpp"
#include "tasks/TaskInfo.hpp"


void TaskMonitor::taskCreated(Task *task, Task *parent) const
{
	assert(task != nullptr);

	TaskStatistics *taskStatistics = task->getTaskStatistics();
	assert(taskStatistics != nullptr);

	// Initialize attributes of the new task
	size_t cost = task->getCost();
	taskStatistics->setCost(cost);

	if (parent != nullptr) {
		TaskStatistics *parentStatistics = parent->getTaskStatistics();
		assert(parentStatistics != nullptr);

		parentStatistics->increaseNumChildrenAlive();

		taskStatistics->setAncestorHasTimePrediction(
			parentStatistics->hasTimePrediction() ||
			parentStatistics->ancestorHasTimePrediction()
		);
	}

	// If the task is a taskfor collaborator, no need to predict anything
	if (task->isTaskforCollaborator()) {
		return;
	}

	// Predict metrics using past data
	TasktypeData *tasktypeData = task->getTasktypeData();
	if (tasktypeData != nullptr) {
		// Predict timing metrics
		TasktypeStatistics &tasktypeStatistics = tasktypeData->getTasktypeStatistics();
		double timePrediction = tasktypeStatistics.getTimingPrediction(cost);
		if (timePrediction != PREDICTION_UNAVAILABLE) {
			taskStatistics->setTimePrediction(timePrediction);
		}

		// Predict hardware counter metrics
		size_t numEnabledCounters = HardwareCounters::getNumEnabledCounters();
		for (size_t i = 0; i < numEnabledCounters; ++i) {
			double counterPrediction = tasktypeStatistics.getCounterPrediction(i, cost);
			if (counterPrediction != PREDICTION_UNAVAILABLE) {
				taskStatistics->setCounterPrediction(i, counterPrediction);
			}
		}

		// Set the task's tasktype statistics for future references
		taskStatistics->setTasktypeStatistics(&(tasktypeStatistics));
	} else if (task->getLabel() == "main") {
		// Create mockup statistics for the main task
		TaskInfo::registerTaskInfo(task->getTaskInfo());

		tasktypeData = task->getTasktypeData();
		assert(tasktypeData != nullptr);

		TasktypeStatistics &tasktypeStatistics = tasktypeData->getTasktypeStatistics();
		taskStatistics->setTasktypeStatistics(&(tasktypeStatistics));
	}
}

void TaskMonitor::taskReinitialized(Task *task) const
{
	assert(task != nullptr);
	assert(task->isTaskfor());

	TaskStatistics *taskStatistics = task->getTaskStatistics();
	assert(taskStatistics != nullptr);

	taskStatistics->reinitialize();

	Task *parent = task->getParent();
	if (parent != nullptr) {
		// NOTE: In the future this assert might need to disappear. For now,
		// this function is only used to reinitialize taskfor collaborators,
		// and when doing that, we need to increase the counter of child tasks
		// so that we can 'markAsFinished' the source @ taskFinished
		assert(parent->isTaskforSource());

		TaskStatistics *parentStatistics = parent->getTaskStatistics();
		assert(parentStatistics != nullptr);

		parentStatistics->increaseNumChildrenAlive();
	}
}

void TaskMonitor::taskStarted(Task *task, monitoring_task_status_t execStatus) const
{
	assert(task != nullptr);

	TaskStatistics *taskStatistics = task->getTaskStatistics();
	assert(taskStatistics != nullptr);

	// Start recording time for the new execution status
	monitoring_task_status_t oldStatus = taskStatistics->startTiming(execStatus);

	// If the task is not a taskfor collaborator, and this is the first time it
	// becomes ready, increase the cost accumulations used to infer predictions.
	// Only if this task doesn't have an ancestor that is already taken into account
	if (!task->isTaskforCollaborator() && !taskStatistics->ancestorHasTimePrediction()) {
		if (oldStatus == null_status && execStatus == ready_status) {
			TasktypeStatistics *tasktypeStatistics = taskStatistics->getTasktypeStatistics();
			assert(tasktypeStatistics != nullptr);

			if (taskStatistics->hasTimePrediction()) {
				size_t cost = taskStatistics->getCost();
				tasktypeStatistics->increaseAccumulatedCost(cost);
				tasktypeStatistics->increaseNumAccumulatedInstances();
			} else {
				tasktypeStatistics->increaseNumPredictionlessInstances();
			}
		}
	}
}

void TaskMonitor::taskCompletedUserCode(Task *task) const
{
	assert(task != nullptr);

	if (task->isTaskforCollaborator()) {
		return;
	}

	TaskStatistics *taskStatistics = task->getTaskStatistics();
	assert(taskStatistics != nullptr);

	// If an ancestor has a prediction, it is taken into account for the
	// accumulation of cost of its tasktype. If that's the case, save this
	// task's elapsed time for the following reasons:
	// 1) Save it in the tasktype statistics of the ancestor
	//    - So that this time can be subtracted from the accumulation of cost
	//      in order to have preciser predictions
	// 2) Save it in the statistics of the ancestor
	//    - So that when the ancestor finishes its execution, this time can be
	//      decreased from the saved time in 1), since the accumulation of cost
	//      will be decreased and we won't need it anymore
	if (taskStatistics->ancestorHasTimePrediction()) {
		Task *ancestor = task->getParent();
		while (ancestor != nullptr) {
			TaskStatistics *ancestorStatistics = ancestor->getTaskStatistics();
			if (!ancestorStatistics->ancestorHasTimePrediction()) {
				// If this ancestor doesn't have an ancestor with predictions, it
				// is the ancestor we're looking for, the one with the prediction
				assert(ancestorStatistics->hasTimePrediction());

				TasktypeStatistics *ancestorTasktypeStatistics = ancestorStatistics->getTasktypeStatistics();
				assert(ancestorTasktypeStatistics != nullptr);

				// Add the elapsed execution time of the task in the ancestor
				// and its tasktype statistics
				size_t elapsed = taskStatistics->getChronoTicks(executing_status);
				ancestorStatistics->increaseCompletedTime(elapsed);
				ancestorTasktypeStatistics->increaseCompletedTime(elapsed);

				break;
			} else {
				ancestor = ancestor->getParent();
			}
		}
	} else if (!taskStatistics->hasTimePrediction()) {
		// If this task has no ancestor with a prediction and it has no prediction,
		// decrease the number of predictionless instances from its tasktype
		TasktypeStatistics *tasktypeStatistics = taskStatistics->getTasktypeStatistics();
		assert(tasktypeStatistics != nullptr);

		tasktypeStatistics->decreaseNumPredictionlessInstances();
	}
}

void TaskMonitor::taskFinished(Task *task) const
{
	assert(task != nullptr);

	TaskStatistics *taskStatistics = task->getTaskStatistics();
	assert(taskStatistics != nullptr);

	// Stop timing for the task
	taskStatistics->stopTiming();

	// NOTE: Special case, for taskfor sources, when the task is finished it
	// also completes user code execution, thus we treat it here
	if (task->isTaskforSource()) {
		taskCompletedUserCode(task);
	}

	// Backpropagate the following actions for the current task and any ancestor
	// that finishes its execution following the finishing of the current task:
	// 1) Accumulate its statistics into its tasktype statistics
	// 2) If there is no ancestor with prediction but the task itself has a
	//    prediction, subtract the time saved @ taskCompletedUserCode (1) of
	//    children tasks of this task from the time saved (2) of this task's
	//    tasktype statistics
	// 3) Accumulate this task's elapsed time and its children elapsed time
	//    into the parent task if it exists
	// NOTE: If the task is a taskfor collaborator, only perform step 3)
	while (taskStatistics->markAsFinished()) {
		assert(!taskStatistics->getNumChildrenAlive());

		// If the task is a taskfor source or a normal task, aggregate
		// timing statistics and counters into its tasktype
		if (!task->isTaskforCollaborator()) {
			TasktypeStatistics *tasktypeStatistics = taskStatistics->getTasktypeStatistics();
			TaskHardwareCounters &taskCounters = task->getHardwareCounters();
			assert(tasktypeStatistics != nullptr);

			// 1)
			tasktypeStatistics->accumulateStatisticsAndCounters(taskStatistics, taskCounters);

			// 2)
			if (!taskStatistics->ancestorHasTimePrediction() && taskStatistics->hasTimePrediction()) {
				tasktypeStatistics->decreaseCompletedTime(taskStatistics->getCompletedTime());
				tasktypeStatistics->decreaseAccumulatedCost(taskStatistics->getCost());
				tasktypeStatistics->decreaseNumAccumulatedInstances();
			}
		}

		// 3)
		Task *parent = task->getParent();
		if (parent != nullptr) {
			TaskStatistics *parentStatistics = parent->getTaskStatistics();
			assert(parentStatistics != nullptr);

			// Accumulate statistics into the parent and follow chain of ancestors
			parentStatistics->accumulateChildrenStatistics(taskStatistics);
			taskStatistics = parentStatistics;
			task = parent;
		} else {
			break;
		}
	}
}

void TaskMonitor::displayStatistics(std::stringstream &stream) const
{
	timeval finalTimestamp;
	gettimeofday(&finalTimestamp, nullptr);

	// Elapsed time in milliseconds
	double elapsedTime = ((finalTimestamp.tv_sec - _initialTimestamp.tv_sec) * 1000.0);
	elapsedTime += ((finalTimestamp.tv_usec - _initialTimestamp.tv_usec) / 1000.0);

	stream << std::left << std::fixed << std::setprecision(5) << "\n";
	stream << "+-----------------------------+\n";
	stream << "|       TASK STATISTICS       |\n";
	stream << "+-----------------------------+\n";

	TaskInfo::processAllTasktypes(
		[&](const std::string &taskLabel, const std::string &, TasktypeData &tasktypeData) {
			TasktypeStatistics &tasktypeStatistics = tasktypeData.getTasktypeStatistics();

			// Display monitoring-related statistics
			size_t numInstances = tasktypeStatistics.getTimingNumInstances();
			if (numInstances) {
				double averageNormalizedCost = tasktypeStatistics.getTimingRollingAverage();
				double stddevNormalizedCost = tasktypeStatistics.getTimingStddev();
				double predictionAccuracy = tasktypeStatistics.getTimingAccuracy();
				double effectiveParallelism = tasktypeStatistics.getAccumulatedTime() / elapsedTime;

				std::string typeLabel = taskLabel + " (" + std::to_string(numInstances) + ")";
				std::string accur = "NA";

				// Make sure there was at least one prediction to report accuracy
				if (!std::isnan(predictionAccuracy)) {
					std::stringstream accuracyStream;
					accuracyStream << std::setprecision(2) << std::fixed << predictionAccuracy << "%";
					accur = accuracyStream.str();
				}

				stream <<
					std::setw(7)  << "STATS"                    << " " <<
					std::setw(12) << "MONITORING"               << " " <<
					std::setw(30) << "TASK-TYPE (INSTANCES)"    << " " <<
					std::setw(25) << typeLabel                  << "\n";
				stream <<
					std::setw(7)  << "STATS"                    << " "   <<
					std::setw(12) << "MONITORING"               << " "   <<
					std::setw(30) << "NORMALIZED COST"          << " "   <<
					std::setw(25) << "AVG / STDEV"              << " "   <<
					std::setw(10) << averageNormalizedCost      << " / " <<
					std::setw(10) << stddevNormalizedCost       << "\n";
				stream <<
					std::setw(7)  << "STATS"                    << " " <<
					std::setw(12) << "MONITORING"               << " " <<
					std::setw(30) << "NORMALIZED COST"          << " " <<
					std::setw(25) << "PREDICTION ACCURACY"      << " " <<
					std::setw(10) << accur                      << "\n";
				stream <<
					std::setw(7)  << "STATS"                    << " " <<
					std::setw(12) << "MONITORING"               << " " <<
					std::setw(30) << "EFFECTIVE PARALLELISM"    << " " <<
					std::setw(25) << "RAW DATA"                 << " " <<
					std::setw(10) << effectiveParallelism       << "\n";
			}

			// Display hardware counters related statistics
			const std::vector<HWCounters::counters_t> &enabledCounters = HardwareCounters::getEnabledCounters();
			for (size_t id = 0; id < enabledCounters.size(); ++id) {
				HWCounters::counters_t eventType = enabledCounters[id];
				numInstances = tasktypeStatistics.getCounterNumInstances(id);
				if (numInstances) {
					// Get statistics
					double counterSum = tasktypeStatistics.getCounterSum(id);
					double counterAvg = tasktypeStatistics.getCounterAverage(id);
					double counterStddev = tasktypeStatistics.getCounterStddev(id);
					double counterAccuracy = tasktypeStatistics.getCounterAccuracy(id);

					// Make sure there was at least one prediction to report accuracy
					std::string accur = "NA";
					if (!std::isnan(counterAccuracy)) {
						std::stringstream accuracyStream;
						accuracyStream << std::setprecision(2) << std::fixed << counterAccuracy << "%";
						accur = accuracyStream.str();
					}

					// Process events that must be in KB
					if (eventType == HWCounters::HWC_PQOS_MON_EVENT_L3_OCCUP ||
						eventType == HWCounters::HWC_PQOS_MON_EVENT_LMEM_BW  ||
						eventType == HWCounters::HWC_PQOS_MON_EVENT_RMEM_BW
					) {
						counterAvg /= 1024.0;
						counterStddev /= 1024.0;
						counterSum /= 1024.0;
					}

					stream <<
						std::setw(7)  << "STATS"                                    << " "   <<
						std::setw(12) << "HWCOUNTERS"                               << " "   <<
						std::setw(30) << HWCounters::counterDescriptions[eventType] << " "   <<
						std::setw(25) << "SUM / AVG / STDEV"                        << " "   <<
						std::setw(15) << counterSum                                 << " / " <<
						std::setw(15) << counterAvg                                 << " / " <<
						std::setw(15) << counterStddev                              << "\n";
					stream <<
						std::setw(7)  << "STATS"                                    << " "   <<
						std::setw(12) << "HWCOUNTERS"                               << " "   <<
						std::setw(30) << HWCounters::counterDescriptions[eventType] << " "   <<
						std::setw(25) << "PREDICTION ACCURACY"                      << " "   <<
						std::setw(10) << accur                                      << "\n";
				}
			}

			// Break after each tasktype
			stream << "+-----------------------------+\n";
		}
	);
}
