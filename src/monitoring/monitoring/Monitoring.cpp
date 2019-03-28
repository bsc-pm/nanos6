
#include "Monitoring.hpp"


EnvironmentVariable<bool> Monitoring::_enabled("NANOS6_MONITORING_ENABLE", true);
Monitoring *Monitoring::_monitor;


//    MONITORING    //

void Monitoring::initialize()
{
	if (_enabled) {
		// Create the monitoring module
		if (_monitor == nullptr) {
			_monitor = new Monitoring();
		}
		
		// Initialize the task monitoring module
		TaskMonitor::initialize();
	}
}

void Monitoring::shutdown()
{
	if (_enabled) {
		// Propagate shutdown to the task monitoring module
		TaskMonitor::shutdown();
		
		// Destroy the monitoring module
		if (_monitor != nullptr) {
			delete _monitor;
		}
		
		_enabled.setValue(false);
	}
}

bool Monitoring::isEnabled()
{
	return _enabled;
}


//    TASKS    //

void Monitoring::taskCreated(Task *task)
{
	if (_enabled) {
		assert(task != nullptr);
		TaskStatistics *taskStatistics = task->getTaskStatistics();
		assert(taskStatistics != nullptr);
		
		// Retrieve information about the task
		const std::string &label = task->getLabel();
		size_t cost = (task->hasCost() ? task->getCost() : DEFAULT_COST);
		
		// Create task statistic structures
		TaskMonitor::taskCreated(taskStatistics, label, cost);
	}
}

void Monitoring::taskChangedStatus(Task *task, monitoring_task_status_t newStatus)
{
	if (_enabled) {
		assert(task != nullptr);
		
		// Start timing for the appropriate stopwatch
		TaskMonitor::startTiming(task->getTaskStatistics(), newStatus);
	}
}

void Monitoring::taskFinished(Task *task)
{
	if (_enabled) {
		assert(task != nullptr);
		
		// Mark task as completely executed
		TaskMonitor::stopTiming(task->getTaskStatistics());
	}
}

