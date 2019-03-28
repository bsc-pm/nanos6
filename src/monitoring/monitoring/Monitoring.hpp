#ifndef MONITORING_HPP
#define MONITORING_HPP

#include "TaskMonitor.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "tasks/Task.hpp"


class Monitoring {

private:
	
	//! Whether monitoring has to be performed or not
	static EnvironmentVariable<bool> _enabled;
	
	// The "monitor", singleton instance
	static Monitoring *_monitor;
	
	
private:
	
	inline Monitoring()
	{
	}
	
	
public:
	
	// Delete copy and move constructors/assign operators
	Monitoring(Monitoring const&) = delete;            // Copy construct
	Monitoring(Monitoring&&) = delete;                 // Move construct
	Monitoring& operator=(Monitoring const&) = delete; // Copy assign
	Monitoring& operator=(Monitoring &&) = delete;     // Move assign
	
	
	//    MONITORING    //
	
	//! \brief Initialize monitoring
	static void initialize();
	
	//! \brief Shutdown monitoring
	static void shutdown();
	
	//! \brief Whether monitoring is enabled
	static bool isEnabled();
	
	
	//    TASKS    //
	
	//! \brief Gather basic information about a task when it is created
	//! \param task The task to gather information about
	static void taskCreated(Task *task);
	
	//! \brief Propagate monitoring operations after a task has changed its
	//! execution status
	//! \param task The task that's changing status
	//! \param newStatus The new execution status of the task
	static void taskChangedStatus(Task *task, monitoring_task_status_t newStatus);
	
	//! \brief Propagate monitoring operations after a task has finished
	//! \param task The task that has finished
	static void taskFinished(Task *task);
	
};

#endif // MONITORING_HPP
