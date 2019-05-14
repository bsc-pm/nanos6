/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "WorkloadPredictor.hpp"


WorkloadPredictor *WorkloadPredictor::_predictor;


void WorkloadPredictor::taskCreated(
	TaskStatistics *taskStatistics,
	TaskPredictions *taskPredictions
) {
	assert(_predictor != nullptr);
	assert(taskStatistics != nullptr);
	assert(taskPredictions != nullptr);
	
	// Account the new task as instantiated workload
	_predictor->increaseInstances(instantiated_load);
	
	// Since the cost of a task includes children tasks, if an ancestor had
	// a prediction, do not take this task into account for workloads
	if (!taskPredictions->ancestorHasPrediction() && taskPredictions->hasPrediction()) {
		const std::string &label = taskStatistics->getLabel();
		size_t cost              = taskStatistics->getCost();
		_predictor->increaseWorkload(instantiated_load, label, cost);
	}
}

void WorkloadPredictor::taskChangedStatus(
	TaskStatistics *taskStatistics,
	TaskPredictions *taskPredictions,
	monitoring_task_status_t oldStatus,
	monitoring_task_status_t newStatus
) {
	assert(_predictor != nullptr);
	assert(taskStatistics != nullptr);
	assert(taskPredictions != nullptr);
	
	const std::string &label = taskStatistics->getLabel();
	size_t cost              = taskStatistics->getCost();
	workload_t decreaseLoad  = WorkloadPredictor::getLoadId(oldStatus);
	workload_t increaseLoad  = WorkloadPredictor::getLoadId(newStatus);
	
	if (decreaseLoad != null_workload) {
		_predictor->decreaseInstances(decreaseLoad);
		if (!taskPredictions->ancestorHasPrediction() && taskPredictions->hasPrediction()) {
			_predictor->decreaseWorkload(decreaseLoad, label, cost);
		}
	}
	
	if (increaseLoad != null_workload) {
		_predictor->increaseInstances(increaseLoad);
		if (!taskPredictions->ancestorHasPrediction() && taskPredictions->hasPrediction()) {
			_predictor->increaseWorkload(increaseLoad, label, cost);
		}
	}
}

void WorkloadPredictor::taskCompletedUserCode(
	TaskStatistics *taskStatistics,
	TaskPredictions *taskPredictions
) {
	assert(_predictor != nullptr);
	assert(taskStatistics != nullptr);
	assert(taskPredictions != nullptr);
	
	// If the task's time is taken into account by an ancestor task (an ancestor has a
	// prediction), when this task finishes we subtract the elapsed time from workloads
	if (taskPredictions->ancestorHasPrediction()) {
		// Find the ancestor who had accounted this task's cost
		TaskPredictions *ancestorPredictions = taskPredictions->getParentPredictions();
		while (!ancestorPredictions->hasPrediction()) {
			ancestorPredictions = ancestorPredictions->getParentPredictions();
		}
		
		// Aggregate the elapsed ticks of the task in the ancestor. When the
		// ancestor has finished, the aggregation will be subtracted from workloads
		size_t elapsed =
			taskStatistics->getChronoTicks(executing_status) +
			taskStatistics->getChronoTicks(runtime_status);
		ancestorPredictions->increaseChildCompletionTimes(elapsed);
		
		// Aggregate the time in workloads also, to obtain better predictions
		_predictor->increaseTaskCompletionTimes(elapsed);
	}
}

void WorkloadPredictor::taskFinished(
	TaskStatistics *taskStatistics,
	TaskPredictions *taskPredictions,
	monitoring_task_status_t oldStatus,
	int ancestorsUpdated
) {
	assert(_predictor != nullptr);
	assert(taskStatistics != nullptr);
	assert(taskPredictions != nullptr);
	assert(ancestorsUpdated >= 0);
	
	// Follow the chain of ancestors and keep updating finished tasks
	while (ancestorsUpdated > 0) {
		assert(!taskStatistics->getAliveChildren());
		const std::string &label = taskStatistics->getLabel();
		size_t cost              = taskStatistics->getCost();
		size_t childTimes        = taskPredictions->getChildCompletionTimes();
		workload_t decreaseLoad  = WorkloadPredictor::getLoadId(oldStatus);
		
		// Decrease workloads by task's statistics
		if (decreaseLoad != null_workload) {
			_predictor->decreaseInstances(decreaseLoad);
			if (!taskPredictions->ancestorHasPrediction() && taskPredictions->hasPrediction()) {
				_predictor->decreaseWorkload(decreaseLoad, label, cost);
			}
		}
		
		// Increase workloads by task's statistics
		_predictor->increaseInstances(finished_load);
		if (!taskPredictions->ancestorHasPrediction() && taskPredictions->hasPrediction()) {
			_predictor->increaseWorkload(finished_load, label, cost);
		}
		
		// Decrease the task completion times, as the task has finished
		_predictor->decreaseTaskCompletionTimes(childTimes);
		
		// Follow the chain of ancestors
		taskStatistics  = taskStatistics->getParentStatistics();
		taskPredictions = taskPredictions->getParentPredictions();
		
		// Decrease the number of ancestors to update
		--ancestorsUpdated;
		
		if (taskStatistics == nullptr || taskPredictions == nullptr) {
			break;
		}
	}
}

double WorkloadPredictor::getPredictedWorkload(workload_t loadId)
{
	assert(_predictor != nullptr);
	
	double totalTime = 0.0;
	for (auto const &it : _predictor->_workloads) {
		totalTime += (
			it.second->getAccumulatedCost(loadId) *
			TaskMonitor::getAverageTimePerUnitOfCost(it.first)
		);
	}
	
	return totalTime;
}

size_t WorkloadPredictor::getTaskCompletionTimes()
{
	assert(_predictor != nullptr);
	
	return _predictor->_taskCompletionTimes.load();
}
