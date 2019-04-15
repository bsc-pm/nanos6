/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef PQOS_THREAD_HARDWARE_COUNTERS_MONITOR_HPP
#define PQOS_THREAD_HARDWARE_COUNTERS_MONITOR_HPP

#include <iomanip>
#include <iostream>
#include <pqos.h>
#include <sstream>

#include "ThreadHardwareCounters.hpp"


class ThreadHardwareCountersMonitor {

private:
	
	//! The singleton instance
	static ThreadHardwareCountersMonitor *_monitor;
	
	
private:
	
	inline ThreadHardwareCountersMonitor()
	{
	}
	
	
public:
	
	// Delete copy and move constructors/assign operators
	ThreadHardwareCountersMonitor(ThreadHardwareCountersMonitor const&) = delete;            // Copy construct
	ThreadHardwareCountersMonitor(ThreadHardwareCountersMonitor&&) = delete;                 // Move construct
	ThreadHardwareCountersMonitor& operator=(ThreadHardwareCountersMonitor const&) = delete; // Copy assign
	ThreadHardwareCountersMonitor& operator=(ThreadHardwareCountersMonitor &&) = delete;     // Move assign
	
	
	//! \brief Initialization of the thread hardware counter monitoring module
	static inline void initialize()
	{
		// Create the monitoring module
		if (_monitor == nullptr) {
			_monitor = new ThreadHardwareCountersMonitor();
		}
	}
	
	//! \brief Shutdown of the thread hardware counter monitoring module
	static inline void shutdown()
	{
		// Destroy the monitoring module
		if (_monitor != nullptr) {
			delete _monitor;
		}
	}
	
	//! \brief Display thread hardware counter statistics
	//! \param stream The output stream
	static inline void displayStatistics(std::stringstream &)
	{
	}
	
	//! \brief Initialize hardware counter monitoring for a thread
	//! \param threadCounters The thread's hardware counter statistics
	//! \param monitoredEvents The PQoS events that must be monitored
	static void initializeThread(ThreadHardwareCounters *threadCounters, pqos_mon_event monitoredEvents);
	
	//! \brief Shutdown hardware counter monitoring for a thread
	//! \param threadCounters The thread's hardware counter statistics
	static void shutdownThread(ThreadHardwareCounters *threadCounters);
	
};

#endif // PQOS_THREAD_HARDWARE_COUNTERS_MONITOR_HPP
