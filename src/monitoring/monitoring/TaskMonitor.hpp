#ifndef TASK_MONITOR_HPP
#define TASK_MONITOR_HPP

#include <string>

#include "TaskStatistics.hpp"
#include "lowlevel/SpinLock.hpp"


class TaskMonitor {

private:
	
	// The monitor singleton instance
	static TaskMonitor *_monitor;
	
	
private:
	
	inline TaskMonitor()
	{
	}
	
	
public:
	
	// Delete copy and move constructors/assign operators
	TaskMonitor(TaskMonitor const&) = delete;            // Copy construct
	TaskMonitor(TaskMonitor&&) = delete;                 // Move construct
	TaskMonitor& operator=(TaskMonitor const&) = delete; // Copy assign
	TaskMonitor& operator=(TaskMonitor &&) = delete;     // Move assign
	
	
	//! \brief Initialize task monitoring
	static inline void initialize()
	{
		// Create the monitoring module
		if (_monitor == nullptr) {
			_monitor = new TaskMonitor();
		}
	}
	
	//! \brief Shutdown task monitoring
	static inline void shutdown()
	{
		// Destroy the monitoring module
		if (_monitor != nullptr) {
			delete _monitor;
		}
	}
	
	//! \brief Initialize a task's monitoring statistics
	//! \param taskStatistics The task's statistics
	//! \param label The tasktype
	//! \param cost The task's computational cost
	static void taskCreated(
		TaskStatistics *taskStatistics,
		const std::string &label,
		size_t cost
	);
	
	//! \brief Start time monitoring for a task
	//! \param taskStatistics The task's statistics
	//! \param execStatus The timing status to start
	//! \return The status before the change
	static monitoring_task_status_t startTiming(TaskStatistics *taskStatistics, monitoring_task_status_t execStatus);
	
	//! \brief Stop time monitoring for a task
	//! \param taskStatistics The task's statistics
	//! \return The status before the change
	static monitoring_task_status_t stopTiming(TaskStatistics *taskStatistics);
	
};

#endif // TASK_MONITOR_HPP
