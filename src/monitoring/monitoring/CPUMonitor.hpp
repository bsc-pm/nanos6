/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_MONITOR_HPP
#define CPU_MONITOR_HPP

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "CPUStatistics.hpp"
#include "executors/threads/CPU.hpp"
#include "executors/threads/CPUManager.hpp"


class CPUMonitor {

private:
	
	//! The vector of CPU statistics, one per compute place
	CPUStatistics *_cpuStatistics;
	
	//! The amount of CPUs of the system
	size_t _numCPUs;
	
	//! The singleton instance for the monitor of statistics
	static CPUMonitor *_monitor;
	
	
private:
	
	inline CPUMonitor()
	{
	}
	
	
public:
	
	// Delete copy and move constructors/assign operators
	CPUMonitor(CPUMonitor const&) = delete;            // Copy construct
	CPUMonitor(CPUMonitor&&) = delete;                 // Move construct
	CPUMonitor& operator=(CPUMonitor const&) = delete; // Copy assign
	CPUMonitor& operator=(CPUMonitor &&) = delete;     // Move assign
	
	
	//    MONITOR    //
	
	//! \brief Initialize CPU monitoring
	static void initialize();
	
	//! \brief Shutdown CPU monitoring
	static void shutdown();
	
	//! \brief Display CPU statistics
	//! \param stream The output stream
	static void displayStatistics(std::stringstream &stream);
	
	
	//    CPU STATUS HANDLING    //
	
	//! \brief Signal that a CPU just became active
	//!
	//! \param virtualCPUId The identifier of the CPU
	static inline void cpuBecomesActive(size_t virtualCPUId)
	{
		assert(_monitor != nullptr);
		
		_monitor->_cpuStatistics[virtualCPUId].cpuBecomesActive();
	}
	
	//! \brief Signal that a CPU just became idle
	//!
	//! \param virtualCPUId The identifier of the CPU
	static inline void cpuBecomesIdle(size_t virtualCPUId)
	{
		assert(_monitor != nullptr);
		
		_monitor->_cpuStatistics[virtualCPUId].cpuBecomesIdle();
	}
	
	//! \brief Retreive the activeness of a CPU
	//!
	//! \param virtualCPUId The identifier of the CPU
	static inline float getActiveness(size_t virtualCPUId)
	{
		assert(_monitor != nullptr);
		
		return _monitor->_cpuStatistics[virtualCPUId].getActiveness();
	}
	
	//! \brief Return the number of CPUs in the system
	static inline size_t getNumCPUs()
	{
		assert(_monitor != nullptr);
		
		return _monitor->_numCPUs;
	}
	
	//! \brief Get the total amount of activeness of all CPUs
	static inline float getTotalActiveness()
	{
		assert(_monitor != nullptr);
		
		float totalActiveness = 0.0;
		for (unsigned short id = 0; id < _monitor->_numCPUs; ++id) {
			totalActiveness += _monitor->_cpuStatistics[id].getActiveness();
		}
		
		return totalActiveness;
	}
	
};

#endif // CPU_MONITOR_HPP
