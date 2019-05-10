
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

