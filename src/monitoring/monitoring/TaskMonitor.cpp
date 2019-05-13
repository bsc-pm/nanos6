
#include "TaskMonitor.hpp"


TaskMonitor *TaskMonitor::_monitor;


void TaskMonitor::taskCreated(
	TaskStatistics  *parentStatistics,
	TaskStatistics  *taskStatistics,
	TaskPredictions *parentPredictions,
	TaskPredictions *taskPredictions,
	const std::string &label,
	size_t cost
) {
	assert(taskStatistics != nullptr);
	assert(taskPredictions != nullptr);
	
	taskStatistics->setParentStatistics(parentStatistics);
	taskStatistics->setLabel(label);
	taskStatistics->setCost(cost);
	if (parentStatistics != nullptr) {
		parentStatistics->increaseAliveChildren();
	}
	
	taskPredictions->setParentPredictions(parentPredictions);
	if (parentPredictions != nullptr) {
		taskPredictions->setAncestorHasPrediction(
			parentPredictions->hasPrediction() ||
			parentPredictions->ancestorHasPrediction()
		);
	}
}

void TaskMonitor::predictTime(TaskPredictions *taskPredictions, const std::string &label, size_t cost)
{
	assert(_monitor != nullptr);
	assert(taskPredictions != nullptr);
	TasktypePredictions *predictions = nullptr;
	
	_monitor->_spinlock.lock();
	
	// Find (or create if unexistent) the tasktype predictions
	tasktype_map_t::iterator it = _monitor->_tasktypeMap.find(label);
	if (it == _monitor->_tasktypeMap.end()) {
		predictions = new TasktypePredictions();
		_monitor->_tasktypeMap.emplace(label, predictions);
	}
	else {
		predictions = it->second;
	}
	
	_monitor->_spinlock.unlock();
	
	assert(predictions != nullptr);
	
	// Predict the execution time of the newly created task
	double timePrediction = predictions->getTimePrediction(cost);
	if (timePrediction != PREDICTION_UNAVAILABLE) {
		taskPredictions->setTimePrediction(timePrediction);
		taskPredictions->setHasPrediction(true);
	}
	
	// Set the task's tasktype prediction reference
	taskPredictions->setTypePredictions(predictions);
}

monitoring_task_status_t TaskMonitor::startTiming(TaskStatistics *taskStatistics, monitoring_task_status_t execStatus)
{
	assert(taskStatistics != nullptr);
	
	return taskStatistics->startTiming(execStatus);
}

monitoring_task_status_t TaskMonitor::stopTiming(TaskStatistics *taskStatistics, TaskPredictions *taskPredictions)
{
	assert(_monitor != nullptr);
	assert(taskStatistics != nullptr);
	assert(taskPredictions != nullptr);
	TaskStatistics      *parentStatistics;
	TaskPredictions     *parentPredictions;
	TasktypePredictions *typePredictions;
	
	// Stop timing for the task
	const monitoring_task_status_t oldStatus = taskStatistics->stopTiming();
	
	// Follow the chain of ancestors and keep updating finished tasks
	bool canAccumulate = taskStatistics->markAsFinished();
	if (canAccumulate) {
		// Update tasktype predictions with this task's statistics
		typePredictions = taskPredictions->getTypePredictions();
		typePredictions->accumulatePredictions(taskStatistics, taskPredictions);
		
		// Retrieve parent statistics and predictions
		parentStatistics  = taskStatistics->getParentStatistics();
		parentPredictions = taskPredictions->getParentPredictions();
		if (parentPredictions != nullptr) {
			typePredictions = parentPredictions->getTypePredictions();
		}
		
		// Backpropagate the update of parent tasks
		if (parentStatistics != nullptr) {
			while (parentStatistics->decreaseAliveChildren()) {
				// Accumulate the task's statistics into the parent task
				parentStatistics->accumulateChildTiming(
					taskStatistics->getChronos(),
					taskStatistics->getChildTimes()
				);
				
				// Update the tasktype predictions with the parent task's statistics
				typePredictions->accumulatePredictions(parentStatistics, parentPredictions);
				
				// Point to the parent's parent task statistics and predictions
				taskStatistics   = parentStatistics;
				taskPredictions  = parentPredictions;
				parentStatistics = taskStatistics->getParentStatistics();
				if (parentStatistics == nullptr) {
					break;
				}
				parentPredictions = taskPredictions->getParentPredictions();
				assert(parentPredictions != nullptr);
				typePredictions = parentPredictions->getTypePredictions();
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
