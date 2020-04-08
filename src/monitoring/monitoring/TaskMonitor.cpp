/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "TaskMonitor.hpp"


TaskMonitor *TaskMonitor::_monitor;


void TaskMonitor::taskCreated(
	TaskStatistics *parentStatistics,
	TaskStatistics *taskStatistics,
	const std::string &label,
	size_t cost
) {
	assert(taskStatistics != nullptr);

	taskStatistics->setParentStatistics(parentStatistics);
	taskStatistics->setLabel(label);
	taskStatistics->setCost(cost);
	if (parentStatistics != nullptr) {
		parentStatistics->increaseAliveChildren();
		taskStatistics->setAncestorHasPrediction(
			parentStatistics->hasPrediction() ||
			parentStatistics->ancestorHasPrediction()
		);
	}
}

void TaskMonitor::predictTime(TaskStatistics *taskStatistics, const std::string &label, size_t cost)
{
	assert(_monitor != nullptr);
	assert(taskStatistics != nullptr);

	TasktypePredictions *predictions = nullptr;

	_monitor->_spinlock.lock();

	// Find (or create if unexistent) the tasktype predictions
	tasktype_map_t::iterator it = _monitor->_tasktypeMap.find(label);
	if (it == _monitor->_tasktypeMap.end()) {
		predictions = new TasktypePredictions();
		_monitor->_tasktypeMap.emplace(label, predictions);
	} else {
		predictions = it->second;
	}

	_monitor->_spinlock.unlock();

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

monitoring_task_status_t TaskMonitor::startTiming(TaskStatistics *taskStatistics, monitoring_task_status_t execStatus)
{
	assert(taskStatistics != nullptr);

	return taskStatistics->startTiming(execStatus);
}

monitoring_task_status_t TaskMonitor::stopTiming(TaskStatistics *taskStatistics, int &ancestorsUpdated)
{
	assert(_monitor != nullptr);
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

double TaskMonitor::getAverageTimePerUnitOfCost(const std::string &label)
{
	_monitor->_spinlock.lock();
	double unitaryTime = _monitor->_tasktypeMap[label]->getAverageTimePerUnitOfCost();
	_monitor->_spinlock.unlock();

	return unitaryTime;
}

void TaskMonitor::insertTimePerUnitOfCost(const std::string &label, double unitaryTime)
{
	assert(_monitor != nullptr);

	TasktypePredictions *predictions = nullptr;

	_monitor->_spinlock.lock();

	// Find (or create if unexistent) the tasktype predictions
	tasktype_map_t::iterator it = _monitor->_tasktypeMap.find(label);
	if (it == _monitor->_tasktypeMap.end()) {
		predictions = new TasktypePredictions();
		_monitor->_tasktypeMap.emplace(label, predictions);
	} else {
		predictions = it->second;
	}

	_monitor->_spinlock.unlock();

	assert(predictions != nullptr);

	// Predict the execution time of the newly created task
	predictions->insertTimePerUnitOfCost(unitaryTime);
}

void TaskMonitor::getAverageTimesPerUnitOfCost(
	std::vector<std::string> &labels,
	std::vector<double> &unitaryTimes
) {
	assert(_monitor != nullptr);

	_monitor->_spinlock.lock();

	// Retrieve all the labels and unitary times
	for (auto const &it : _monitor->_tasktypeMap) {
		if (it.second != nullptr) {
			labels.push_back(it.first);
			unitaryTimes.push_back(it.second->getAverageTimePerUnitOfCost());
		}
	}

	_monitor->_spinlock.unlock();
}
