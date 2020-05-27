/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "TaskMonitor.hpp"
#include "TasktypeStatistics.hpp"
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
			parentStatistics->increaseNumChildrenAlive();
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

	// If the task is not a taskfor collaborator, and this is the first time it
	// becomes ready, increase the cost accumulations used to infer predictions.
	// Only if this task doesn't have an ancestor that is already taken into account
	bool isCollaborator = (task->isTaskfor() && task->isRunnable());
	if (!isCollaborator && !taskStatistics->ancestorHasPrediction()) {
		if ((oldStatus == null_status || oldStatus == pending_status) && execStatus == ready_status) {
			TasktypeStatistics *tasktypeStatistics = taskStatistics->getTasktypeStatistics();
			assert(tasktypeStatistics != nullptr);

			if (taskStatistics->hasPrediction()) {
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
	bool isCollaborator = (task->isTaskfor() && task->isRunnable());
	if (!isCollaborator) {
		if (taskStatistics->ancestorHasPrediction()) {
			TaskStatistics *ancestorStatistics = taskStatistics->getParentStatistics();
			while (ancestorStatistics != nullptr) {
				if (!ancestorStatistics->ancestorHasPrediction()) {
					// If this ancestor doesn't have an ancestor with predictions, it
					// is the ancestor we're looking for, the one with the prediction
					assert(ancestorStatistics->hasPrediction());

					TasktypeStatistics *ancestorTasktypeStatistics = ancestorStatistics->getTasktypeStatistics();
					assert(ancestorTasktypeStatistics != nullptr);

					// Add the elapsed execution time of the task in the ancestor
					// and its tasktype statistics
					size_t elapsed = taskStatistics->getChronoTicks(executing_status);
					ancestorStatistics->increaseCompletedTime(elapsed);
					ancestorTasktypeStatistics->increaseCompletedTime(elapsed);

					break;
				} else {
					ancestorStatistics = ancestorStatistics->getParentStatistics();
				}
			}
		} else if (!taskStatistics->hasPrediction()) {
			// If this task has no ancestor with a prediction and it has no prediction,
			// decrease the number of predictionless instances from its tasktype
			TasktypeStatistics *tasktypeStatistics = taskStatistics->getTasktypeStatistics();
			assert(tasktypeStatistics != nullptr);

			tasktypeStatistics->decreaseNumPredictionlessInstances();
		}
	}
}

void TaskMonitor::taskFinished(Task *task) const
{
	assert(task != nullptr);

	TaskStatistics *taskStatistics = task->getTaskStatistics();
	assert(taskStatistics != nullptr);

	// NOTE: Special case. For taskfor sources, when the task is finished
	// it also completes user code execution, thus we treat it here
	bool isSourceTaskfor = (task->isTaskfor() && !task->isRunnable());
	if (isSourceTaskfor) {
		taskCompletedUserCode(task);
	}

	// Stop timing for the task
	__attribute__((unused)) monitoring_task_status_t oldStatus = taskStatistics->stopTiming();

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

		// Make sure this is not a taskfor collaborator for these steps
		if (!task->isTaskfor() || (task->isTaskfor() && !task->isRunnable())) {
			TasktypeStatistics *tasktypeStatistics = taskStatistics->getTasktypeStatistics();
			assert(tasktypeStatistics != nullptr);

			// 1)
			tasktypeStatistics->accumulateStatistics(taskStatistics);

			// 2)
			if (!taskStatistics->ancestorHasPrediction() && taskStatistics->hasPrediction()) {
				tasktypeStatistics->decreaseCompletedTime(taskStatistics->getCompletedTime());
				tasktypeStatistics->decreaseAccumulatedCost(taskStatistics->getCost());
				tasktypeStatistics->decreaseNumAccumulatedInstances();
			}
		}

		// 3)
		TaskStatistics *parentStatistics = taskStatistics->getParentStatistics();
		if (parentStatistics != nullptr) {
			parentStatistics->accumulateChildrenStatistics(taskStatistics);
			taskStatistics = parentStatistics;
			task = task->getParent();
			assert(task != nullptr);
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
		[&](const std::string &taskLabel, TasktypeData &tasktypeData) {
			TasktypeStatistics &tasktypeStatistics = tasktypeData.getTasktypeStatistics();
			size_t instances = tasktypeStatistics.getNumInstances();
			if (instances) {
				double averageNormalizedCost = tasktypeStatistics.getAverageNormalizedCost();
				double stddevNormalizedCost = tasktypeStatistics.getStddevNormalizedCost();
				double predictionAccuracy = tasktypeStatistics.getPredictionAccuracy();
				double effectiveParallelism = tasktypeStatistics.getAccumulatedTime() / elapsedTime;

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
				stream <<
					std::setw(7)  << "STATS"                    << " " <<
					std::setw(12) << "MONITORING"               << " " <<
					std::setw(26) << "EFFECTIVE PARALLELISM"    << " " <<
					std::setw(10) << effectiveParallelism       << "\n";
				stream << "+-----------------------------+\n";
			}
		}
	);

	stream << "\n";
}
