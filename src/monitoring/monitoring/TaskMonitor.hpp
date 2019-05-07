#ifndef TASK_MONITOR_HPP
#define TASK_MONITOR_HPP

#include <map>
#include <string>

#include "TasktypePredictions.hpp"
#include "lowlevel/SpinLock.hpp"


class TaskMonitor {

private:
	
	typedef std::map< std::string, TasktypePredictions *> tasktype_map_t;
	
	// The monitor singleton instance
	static TaskMonitor *_monitor;
	
	//! Maps TasktypePredictions by task labels
	tasktype_map_t _tasktypeMap;
	
	//! Spinlock that ensures atomic access within the tasktype map
	SpinLock _spinlock;
	
	
private:
	
	inline TaskMonitor() :
		_tasktypeMap(),
		_spinlock()
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
		if (_monitor != nullptr) {
			// Destroy all the task type statistics
			for (auto &it : _monitor->_tasktypeMap) {
				if (it.second != nullptr) {
					delete it.second;
				}
			}
			
			delete _monitor;
		}
	}
	
	//! \brief Initialize a task's monitoring statistics
	//! \param parentStatistics The parent task's statistics
	//! \param taskStatistics The task's statistics
	//! \param parentPredictions The parent task's predictions
	//! \param taskPredictions The task's predictions
	//! \param label The tasktype
	//! \param cost The task's computational cost
	static void taskCreated(
		TaskStatistics  *parentStatistics,
		TaskStatistics  *taskStatistics,
		TaskPredictions *parentPredictions,
		TaskPredictions *taskPredictions,
		const std::string &label,
		size_t cost
	);
	
	//! \brief Predict the execution time of a task
	//! \param taskPredictions The predictions of the task
	//! \param label The tasktype
	//! \param cost The task's computational task
	static void predictTime(TaskPredictions *taskPredictions, const std::string &label, size_t cost);
	
	//! \brief Start time monitoring for a task
	//! \param taskStatistics The task's statistics
	//! \param execStatus The timing status to start
	//! \return The status before the change
	static monitoring_task_status_t startTiming(TaskStatistics *taskStatistics, monitoring_task_status_t execStatus);
	
	//! \brief Stop time monitoring for a task
	//! \param taskStatistics The task's statistics
	//! \return The status before the change
	static monitoring_task_status_t stopTiming(TaskStatistics *taskStatistics, TaskPredictions *taskPredictions);
	
	//! \brief Get an average time per unit of cost value of a tasktype
	//! \param label The tasktype
	static double getAverageTimePerUnitOfCost(const std::string &label);
	
};

#endif // TASK_MONITOR_HPP
