/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "TaskHardwareCountersMonitor.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


TaskHardwareCountersMonitor *TaskHardwareCountersMonitor::_monitor;


void TaskHardwareCountersMonitor::taskCreated(TaskHardwareCounters *taskCounters, const std::string &label, size_t cost)
{
	assert(taskCounters != nullptr);
	
	taskCounters->setLabel(label);
	taskCounters->setCost(cost);
}

void TaskHardwareCountersMonitor::predictTaskCounters(TaskHardwareCountersPredictions *taskPredictions, const std::string &label, size_t cost)
{
	assert(_monitor != nullptr);
	assert(taskPredictions != nullptr);
	tasktype_hardware_counters_map_t &tasktypeMap = _monitor->_tasktypeMap;
	TasktypeHardwareCountersPredictions *predictions = nullptr;
	
	_monitor->_spinlock.lock();
	
	// Find (or create if unexistent) the tasktype HW counter predictions
	tasktype_hardware_counters_map_t::iterator it = tasktypeMap.find(label);
	if (it == tasktypeMap.end()) {
		predictions = new TasktypeHardwareCountersPredictions();
		tasktypeMap.emplace(label, predictions);
	}
	else {
		predictions = it->second;
	}
	
	_monitor->_spinlock.unlock();
	
	// Predict the execution time of the newly created task
	assert(predictions != nullptr);
	for (unsigned short counterId = 0; counterId < HWCounters::num_counters; ++counterId) {
		double counterPrediction = predictions->getCounterPrediction((HWCounters::counters_t) counterId, cost);
		if (counterPrediction != PREDICTION_UNAVAILABLE) {
			taskPredictions->setCounterPrediction((HWCounters::counters_t) counterId, counterPrediction);
			taskPredictions->setPredictionAvailable((HWCounters::counters_t) counterId, true);
		}
	}
	
	taskPredictions->setTypePredictions(predictions);
}

void TaskHardwareCountersMonitor::startTaskMonitoring(TaskHardwareCounters *taskCounters, pqos_mon_data *threadData)
{
	assert(taskCounters != nullptr);
	assert(threadData != nullptr);
	
	if (!taskCounters->isCurrentlyMonitoring()) {
		// Poll PQoS events from the current thread only
		int ret = pqos_mon_poll(&threadData, 1);
		FatalErrorHandler::failIf(ret != PQOS_RETVAL_OK, "Error '", ret, "' when polling PQoS events for a task (start/resume)");
		
		// If successfull, save counters when the task starts or resumes execution
		taskCounters->startOrResume(threadData);
	}
}

void TaskHardwareCountersMonitor::stopTaskMonitoring(TaskHardwareCounters *taskCounters, pqos_mon_data *threadData)
{
	assert(taskCounters != nullptr);
	assert(threadData != nullptr);
	
	if (taskCounters->isCurrentlyMonitoring()) {
		// Poll PQoS events from the current thread only
		int ret = pqos_mon_poll(&threadData, 1);
		FatalErrorHandler::failIf(ret != PQOS_RETVAL_OK, "Error '", ret, "' when polling PQoS events for a task (stop/pause)");
		
		// If successfull, save counters when the task stops or pauses
		// execution, and accumulate them
		taskCounters->stopOrPause(threadData);
		taskCounters->accumulateCounters();
	}
}

void TaskHardwareCountersMonitor::taskFinished(TaskHardwareCounters *taskCounters, TaskHardwareCountersPredictions *taskPredictions)
{
	assert(_monitor != nullptr);
	assert(taskPredictions != nullptr);
	
	// Aggregate hardware counters statistics and predictions into tasktype counters
	taskPredictions->getTypePredictions()->accumulateCounters(taskCounters, taskPredictions);
}

void TaskHardwareCountersMonitor::insertCounterValuesPerUnitOfCost(
		const std::string &label,
		std::vector<HWCounters::counters_t> &counterIds,
		std::vector<double> &counterValues
) {
	assert(_monitor != nullptr);
	tasktype_hardware_counters_map_t &tasktypeMap = _monitor->_tasktypeMap;
	TasktypeHardwareCountersPredictions *predictions = nullptr;
	
	// Find (or create if unexistent) the tasktype HW counter predictions
	_monitor->_spinlock.lock();
	
	tasktype_hardware_counters_map_t::iterator it = tasktypeMap.find(label);
	if (it == tasktypeMap.end()) {
		predictions = new TasktypeHardwareCountersPredictions();
		tasktypeMap.emplace(label, predictions);
	}
	else {
		predictions = it->second;
	}
	
	_monitor->_spinlock.unlock();
	
	predictions->insertCounterValuesPerUnitOfCost(counterIds, counterValues);
}

void TaskHardwareCountersMonitor::getAverageCounterValuesPerUnitOfCost(
	std::vector<std::string> &labels,
	std::vector<std::vector<std::pair<HWCounters::counters_t, double>>> &unitaryValues
) {
	assert(_monitor != nullptr);
	
	_monitor->_spinlock.lock();
	
	// Retrieve all the labels and unitary counter ids and values
	short index = 0;
	short numCounters = HWCounters::num_counters;
	for (auto const &it : _monitor->_tasktypeMap) {
		const std::string &label = it.first;
		TasktypeHardwareCountersPredictions *predictions = it.second;
		
		if (predictions != nullptr) {
			labels.push_back(label);
			unitaryValues.push_back(std::vector<std::pair<HWCounters::counters_t, double>>(numCounters));
			for (unsigned short counterId = 0; counterId < numCounters; ++counterId) {
				unitaryValues[index].push_back(
					std::make_pair(
						(HWCounters::counters_t) counterId,
						predictions->getNormalizedCounterRollingAverage((HWCounters::counters_t) counterId)
					)
				);
			}
			index++;
		}
	}
	
	_monitor->_spinlock.unlock();
}
