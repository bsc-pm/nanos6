
#include "TaskMonitor.hpp"


TaskMonitor *TaskMonitor::_monitor;


void TaskMonitor::taskCreated(
	TaskStatistics *taskStatistics,
	const std::string &label,
	size_t cost
) {
	assert(taskStatistics != nullptr);
	
	taskStatistics->setLabel(label);
	taskStatistics->setCost(cost);
}

monitoring_task_status_t TaskMonitor::startTiming(TaskStatistics *taskStatistics, monitoring_task_status_t execStatus)
{
	assert(taskStatistics != nullptr);
	
	return taskStatistics->startTiming(execStatus);
}

monitoring_task_status_t TaskMonitor::stopTiming(TaskStatistics *taskStatistics)
{
	assert(taskStatistics != nullptr);
	
	return taskStatistics->stopTiming();
}

