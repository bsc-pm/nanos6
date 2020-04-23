/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "TaskMonitor.hpp"


void TaskMonitor::taskCreated(
	TaskStatistics *parentStatistics,
	TaskStatistics *taskStatistics,
	const std::string &label,
	size_t cost
) {
	assert(taskStatistics != nullptr);

	// Initialize attributes of the new task
	taskStatistics->setParentStatistics(parentStatistics);
	taskStatistics->setLabel(label);
	taskStatistics->setCost(cost);
	if (parentStatistics != nullptr) {
		parentStatistics->increaseAliveChildren();
		parentStatistics->increaseNumChildren();
		taskStatistics->setAncestorHasPrediction(
			parentStatistics->hasPrediction() ||
			parentStatistics->ancestorHasPrediction()
		);
	}

	// Predict metrics using past data
	TasktypePredictions *predictions = nullptr;

	_spinlock.lock();

	// Find (or create if unexistent) the tasktype predictions
	tasktype_map_t::iterator it = _tasktypeMap.find(label);
	if (it == _tasktypeMap.end()) {
		predictions = new TasktypePredictions();
		_tasktypeMap.emplace(label, predictions);
	} else {
		predictions = it->second;
	}

	_spinlock.unlock();

	assert(predictions != nullptr);

	// Predict the execution time of the newly created task
	double timePrediction = predictions->getTimePrediction(cost);
	if (timePrediction != PREDICTION_UNAVAILABLE) {
		taskStatistics->setTimePrediction(timePrediction);
		taskStatistics->setHasPrediction(true);
	}

	// Set the task's tasktype prediction reference
	taskStatistics->setTypePredictions(predictions);
}

void TaskMonitor::taskReinitialized(TaskStatistics *taskStatistics) const
{
	assert(taskStatistics != nullptr);

	taskStatistics->reinitialize();
}

monitoring_task_status_t TaskMonitor::startTiming(TaskStatistics *taskStatistics, monitoring_task_status_t execStatus) const
{
	assert(taskStatistics != nullptr);

	return taskStatistics->startTiming(execStatus);
}

monitoring_task_status_t TaskMonitor::stopTiming(TaskStatistics *taskStatistics, int &ancestorsUpdated) const
{
	assert(taskStatistics != nullptr);

	TaskStatistics      *parentStatistics;
	TasktypePredictions *typePredictions;

	// Stop timing for the task
	const monitoring_task_status_t oldStatus = taskStatistics->stopTiming();

	// Follow the chain of ancestors and keep updating finished tasks
	bool canAccumulate = taskStatistics->markAsFinished();
	if (canAccumulate) {
		// Increase the number of ancestors that have been updated
		// This is the own task having finished
		++ancestorsUpdated;

		// Update tasktype predictions with this task's statistics
		typePredictions = taskStatistics->getTypePredictions();
		typePredictions->accumulatePredictions(taskStatistics);

		// Retrieve parent statistics and predictions
		parentStatistics  = taskStatistics->getParentStatistics();
		if (parentStatistics != nullptr) {
			typePredictions = parentStatistics->getTypePredictions();
		}

		// Backpropagate the update of parent tasks
		if (parentStatistics != nullptr) {
			while (parentStatistics->decreaseAliveChildren()) {
				// Increase the number of ancestors that have been updated
				++ancestorsUpdated;

				// Accumulate the task's statistics into the parent task
				parentStatistics->accumulateChildTiming(
					taskStatistics->getChronos(),
					taskStatistics->getChildTimes()
				);

				// Update the tasktype predictions with the parent task's statistics
				typePredictions->accumulatePredictions(parentStatistics);

				// Point to the parent's parent task statistics and predictions
				taskStatistics   = parentStatistics;
				parentStatistics = taskStatistics->getParentStatistics();
				if (parentStatistics == nullptr) {
					break;
				}

				typePredictions = parentStatistics->getTypePredictions();
			}
		}
	}

	return oldStatus;
}

void TaskMonitor::taskforCollaboratorEnded(TaskStatistics *collaboratorStatistics, TaskStatistics *taskforStatistics) const
{
	assert(taskforStatistics != nullptr);
	assert(collaboratorStatistics != nullptr);

	// Accumulate the collaborator's statistics into the parent Taskfor
	taskforStatistics->accumulateChildTiming(
		collaboratorStatistics->getChronos(),
		collaboratorStatistics->getChildTimes()
	);
}

double TaskMonitor::getAverageTimePerUnitOfCost(const std::string &label)
{
	_spinlock.lock();
	double unitaryTime = _tasktypeMap[label]->getAverageTimePerUnitOfCost();
	_spinlock.unlock();

	return unitaryTime;
}

void TaskMonitor::insertTimePerUnitOfCost(const std::string &label, double unitaryTime)
{
	TasktypePredictions *predictions = nullptr;

	_spinlock.lock();

	// Find (or create if unexistent) the tasktype predictions
	tasktype_map_t::iterator it = _tasktypeMap.find(label);
	if (it == _tasktypeMap.end()) {
		predictions = new TasktypePredictions();
		_tasktypeMap.emplace(label, predictions);
	} else {
		predictions = it->second;
	}

	_spinlock.unlock();

	assert(predictions != nullptr);

	// Predict the execution time of the newly created task
	predictions->insertTimePerUnitOfCost(unitaryTime);
}

void TaskMonitor::getAverageTimesPerUnitOfCost(
	std::vector<std::string> &labels,
	std::vector<double> &unitaryTimes
) {
	_spinlock.lock();

	// Retrieve all the labels and unitary times
	for (auto const &it : _tasktypeMap) {
		if (it.second != nullptr) {
			labels.push_back(it.first);
			unitaryTimes.push_back(it.second->getAverageTimePerUnitOfCost());
		}
	}

	_spinlock.unlock();
}

void TaskMonitor::displayStatistics(std::stringstream &stream)
{
	stream << std::left << std::fixed << std::setprecision(5) << "\n";
	stream << "+-----------------------------+\n";
	stream << "|       TASK STATISTICS       |\n";
	stream << "+-----------------------------+\n";

	for (auto const &it : _tasktypeMap) {
		int instances = it.second->getInstances();
		if (instances) {
			double avgCost        = it.second->getAverageTimePerUnitOfCost();
			double stdevCost      = it.second->getStdevTimePerUnitOfCost();
			double accuracy       = it.second->getPredictionAccuracy();
			std::string typeLabel = it.first + " (" + std::to_string(instances) + ")";
			std::string accur = "NA";

			// Make sure there was at least one prediction to report accuracy
			if (!std::isnan(accuracy)) {
				std::stringstream accuracyStream;
				accuracyStream << std::setprecision(2) << std::fixed << accuracy << "%";
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
				std::setw(10) << avgCost                    << " / " <<
				std::setw(10) << stdevCost                  << "\n";
			stream <<
				std::setw(7)  << "STATS"                    << " " <<
				std::setw(12) << "MONITORING"               << " " <<
				std::setw(26) << "PREDICTION ACCURACY (%)"  << " " <<
				std::setw(10) << accur                      << "\n";
			stream << "+-----------------------------+\n";
		}
	}
	stream << "\n";
}
