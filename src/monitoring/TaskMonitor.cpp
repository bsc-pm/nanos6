/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "TaskMonitor.hpp"
#include "tasks/TaskInfo.hpp"


void TaskMonitor::taskCreated(Task *task, Task *parent) const
{
	assert(task != nullptr);

	TaskStatistics *taskStatistics = task->getTaskStatistics();
	assert(taskStatistics != nullptr);

	// Initialize attributes of the new task
	TaskStatistics *parentStatistics = nullptr;
	if (parent != nullptr) {
		parentStatistics = parent->getTaskStatistics();
	}
	size_t cost = task->getCost();
	taskStatistics->setParentStatistics(parentStatistics);
	taskStatistics->setCost(cost);

	if (parent != nullptr) {
		if (parentStatistics != nullptr) {
			parentStatistics->increaseAliveChildren();
			parentStatistics->increaseNumChildren();
			taskStatistics->setAncestorHasPrediction(
				parentStatistics->hasPrediction() ||
				parentStatistics->ancestorHasPrediction()
			);
		}
	}

	// Predict metrics using past data
	TasktypeData *tasktypeData = task->getTasktypeData();
	if (tasktypeData != nullptr) {
		TasktypeStatistics &tasktypeStatistics = tasktypeData->getTasktypeStatistics();
		double timePrediction = tasktypeStatistics.getTimePrediction(cost);
		if (timePrediction != PREDICTION_UNAVAILABLE) {
			taskStatistics->setTimePrediction(timePrediction);
			taskStatistics->setHasPrediction(true);
		}

		// Set the task's tasktype statistics for future references
		taskStatistics->setTasktypeStatistics(&(tasktypeStatistics));

		// Since the cost of a task includes children tasks, if an ancestor had
		// a prediction, do not take this task into account for workloads
		if (!taskStatistics->ancestorHasPrediction() && taskStatistics->hasPrediction()) {
			tasktypeStatistics.increaseAccumulatedCost(instantiated_load, cost);
		}
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

	TaskStatistics *taskStatistics = task->getTaskStatistics();
	assert(taskStatistics != nullptr);

	taskStatistics->reinitialize();
}

void TaskMonitor::taskStarted(Task *task, monitoring_task_status_t execStatus) const
{
	assert(task != nullptr);

	TaskStatistics *taskStatistics = task->getTaskStatistics();
	assert(taskStatistics != nullptr);

	// Start recording time for the new execution status
	monitoring_task_status_t oldStatus = taskStatistics->startTiming(execStatus);

	// If the task is not a taskfor, increase/decrease the appropriate
	// cost accumulations used to infer predictions
	if (!task->isTaskfor()) {
		workload_t decreaseLoad = getLoadId(oldStatus);
		workload_t increaseLoad = getLoadId(execStatus);
		size_t cost = taskStatistics->getCost();
		TasktypeStatistics *tasktypeStatistics = taskStatistics->getTasktypeStatistics();
		assert(tasktypeStatistics != nullptr);

		if (decreaseLoad != null_workload) {
			if (!taskStatistics->ancestorHasPrediction() && taskStatistics->hasPrediction()) {
				tasktypeStatistics->decreaseAccumulatedCost(decreaseLoad, cost);
			}
		}

		if (increaseLoad != null_workload) {
			if (!taskStatistics->ancestorHasPrediction() && taskStatistics->hasPrediction()) {
				tasktypeStatistics->increaseAccumulatedCost(increaseLoad, cost);
			}
		}
	}
}

void TaskMonitor::taskFinished(Task *task) const
{
	assert(task != nullptr);

	TaskStatistics *taskStatistics = task->getTaskStatistics();
	assert(taskStatistics != nullptr);

	TasktypeStatistics *tasktypeStatistics = taskStatistics->getTasktypeStatistics();
	assert(tasktypeStatistics != nullptr);

	// Stop timing for the task
	monitoring_task_status_t oldStatus = taskStatistics->stopTiming();

	// If the task is not a taskfor, decrease the appropriate cost
	// accumulations used to infer predictions
	if (!task->isTaskfor()) {
		size_t cost = taskStatistics->getCost();
		workload_t decreaseLoad = getLoadId(oldStatus);
		if (decreaseLoad != null_workload) {
			if (!taskStatistics->ancestorHasPrediction() && taskStatistics->hasPrediction()) {
				tasktypeStatistics->decreaseAccumulatedCost(decreaseLoad, cost);
			}
		}
	}

	// Follow the chain of ancestors and accumulate statistics from finished tasks
	bool canAccumulate = taskStatistics->markAsFinished();
	if (canAccumulate) {
		// Update tasktype statistics with this task's statistics
		tasktypeStatistics->accumulateStatistics(taskStatistics);

		TaskStatistics *parentStatistics = taskStatistics->getParentStatistics();
		if (parentStatistics != nullptr) {
			// Accumulate the tasks statistics into the parent task
			parentStatistics->accumulateChildTiming(
				taskStatistics->getChronos(),
				taskStatistics->getChildTimes()
			);

			// Backpropagate the update of parent tasks
			while (parentStatistics->decreaseAliveChildren()) {
				// If we enter this condition, we are the last child of the
				// parent task, so its statistics can be accumulated
				assert(!parentStatistics->getAliveChildren());

				// Update the tasktype predictions with the parent task's statistics
				tasktypeStatistics = parentStatistics->getTasktypeStatistics();
				tasktypeStatistics->accumulateStatistics(parentStatistics);

				// Point to the parent's parent task statistics
				taskStatistics = parentStatistics;
				parentStatistics = taskStatistics->getParentStatistics();
				if (parentStatistics != nullptr) {
					parentStatistics->accumulateChildTiming(
						taskStatistics->getChronos(),
						taskStatistics->getChildTimes()
					);
				} else {
					break;
				}
			}
		}
	}
}

void TaskMonitor::taskforFinished(Task *task, Task *source) const
{
	assert(task != nullptr);
	assert(source != nullptr);
	assert(source->isTaskfor());

	TaskStatistics *taskforStatistics = task->getTaskStatistics();
	TaskStatistics *sourceStatistics = source->getTaskStatistics();
	assert(taskforStatistics != nullptr);
	assert(sourceStatistics != nullptr);

	// Stop timing for the taskfor
	taskforStatistics->stopTiming();

	// Accumulate the collaborator's statistics into the parent Taskfor
	sourceStatistics->accumulateChildTiming(
		taskforStatistics->getChronos(),
		taskforStatistics->getChildTimes()
	);
}

void TaskMonitor::displayStatistics(std::stringstream &stream) const
{
	stream << std::left << std::fixed << std::setprecision(5) << "\n";
	stream << "+-----------------------------+\n";
	stream << "|       TASK STATISTICS       |\n";
	stream << "+-----------------------------+\n";

	TaskInfo::processAllTasktypes(
		[&](const std::string &taskLabel, TasktypeData &tasktypeData) {
			TasktypeStatistics &tasktypeStatistics = tasktypeData.getTasktypeStatistics();
			size_t instances = tasktypeStatistics.getNumInstances();
			if (instances) {
				double averageNormalizedCost = tasktypeStatistics.getAverageNormalizedCost();
				double stddevNormalizedCost = tasktypeStatistics.getStddevNormalizedCost();
				double predictionAccuracy = tasktypeStatistics.getPredictionAccuracy();

				std::string typeLabel = taskLabel + " (" + std::to_string(instances) + ")";
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
					std::setw(26) << "TASK-TYPE (INSTANCES)"    << " " <<
					std::setw(20) << typeLabel                  << "\n";
				stream <<
					std::setw(7)  << "STATS"                    << " "   <<
					std::setw(12) << "MONITORING"               << " "   <<
					std::setw(26) << "UNITARY COST AVG / STDEV" << " "   <<
					std::setw(10) << averageNormalizedCost      << " / " <<
					std::setw(10) << stddevNormalizedCost       << "\n";
				stream <<
					std::setw(7)  << "STATS"                    << " " <<
					std::setw(12) << "MONITORING"               << " " <<
					std::setw(26) << "PREDICTION ACCURACY (%)"  << " " <<
					std::setw(10) << accur                      << "\n";
				stream << "+-----------------------------+\n";
			}
		}
	);

	stream << "\n";
}
