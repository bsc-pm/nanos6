/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef PQOS_HARDWARE_COUNTERS_HPP
#define PQOS_HARDWARE_COUNTERS_HPP

#include "TaskHardwareCountersMonitor.hpp"
#include "ThreadHardwareCountersMonitor.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "tasks/Task.hpp"


class HardwareCounters {

private:
	
	//! Whether hardware counter monitoring is enabled
	static EnvironmentVariable<bool> _enabled;
	
	//! Whether the verbose mode is enabled
	static EnvironmentVariable<bool> _verbose;
	
	//! The file where output must be saved when verbose mode is enabled
	static EnvironmentVariable<std::string> _outputFile;
	
	//! The singleton instance
	static HardwareCounters *_monitor;
	
	
private:
	
	inline HardwareCounters()
	{
	}
	
	
public:
	
	// Delete copy and move constructors/assign operators
	HardwareCounters(HardwareCounters const&) = delete;            // Copy construct
	HardwareCounters(HardwareCounters&&) = delete;                 // Move construct
	HardwareCounters& operator=(HardwareCounters const&) = delete; // Copy assign
	HardwareCounters& operator=(HardwareCounters &&) = delete;     // Move assign
	
	
	//    HARDWARE COUNTERS    //
	
	//! \brief Initialization of the hardware counter monitoring module
	static void initialize();
	
	//! \brief Shutdown of the hardware counter monitoring module
	static void shutdown();
	
	//! \brief Display hardware counter statistics
	static void displayStatistics();
	
	//! \brief Whether monitoring is enabled
	static bool isEnabled();
	
	
	//    TASKS    //
	
	//! \brief Gather basic information about a task when it is created
	//! \param[in] task The task to gather information about
	static void taskCreated(Task *task);
	
	//! \brief Start/resume hardware counter monitoring for a task
	//! \param task The task to start monitoring for
	static void startTaskMonitoring(Task *task);
	
	//! \brief Stop/pause hardware counter monitoring for a task and aggregate
	//! the current thread's counter into the task's counters
	//! \param task The task to start monitoring for
	static void stopTaskMonitoring(Task *task);
	
	//! \brief Finish hardware counter monitoring for a task
	//! \param task The task that has finished
	static void taskFinished(Task *task);
	
	
	//    THREADS    //
	
	//! \brief Initialize hardware counter monitoring for the current thread
	static void initializeThread();
	
	//! \brief Shutdown hardware counter monitoring for the current thread
	static void shutdownThread();
	
};

#endif // PQOS_HARDWARE_COUNTERS_HPP
