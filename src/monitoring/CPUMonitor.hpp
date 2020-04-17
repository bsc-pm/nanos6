/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
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
	static CPUStatistics *_cpuStatistics;

	//! The amount of CPUs of the system
	static size_t _numCPUs;

public:

	//    MONITOR    //

	//! \brief Initialize CPU monitoring
	static void initialize();

	//! \brief Shutdown CPU monitoring
	static void shutdown();

	//! \brief Display CPU statistics
	//!
	//! \param[in,out] stream The output stream
	static void displayStatistics(std::stringstream &stream);


	//    CPU STATUS HANDLING    //

	//! \brief Signal that a CPU just became active
	//!
	//! \param[in] virtualCPUId The identifier of the CPU
	static inline void cpuBecomesActive(int virtualCPUId)
	{
		_cpuStatistics[virtualCPUId].cpuBecomesActive();
	}

	//! \brief Signal that a CPU just became idle
	//!
	//! \param[in] virtualCPUId The identifier of the CPU
	static inline void cpuBecomesIdle(int virtualCPUId)
	{
		_cpuStatistics[virtualCPUId].cpuBecomesIdle();
	}

	//! \brief Retreive the activeness of a CPU
	//!
	//! \param[in] virtualCPUId The identifier of the CPU
	static inline float getActiveness(int virtualCPUId)
	{
		return _cpuStatistics[virtualCPUId].getActiveness();
	}

	//! \brief Return the number of CPUs in the system
	static inline size_t getNumCPUs()
	{
		return _numCPUs;
	}

	//! \brief Get the total amount of activeness of all CPUs
	static inline float getTotalActiveness()
	{
		float totalActiveness = 0.0;
		for (unsigned short id = 0; id < _numCPUs; ++id) {
			totalActiveness += _cpuStatistics[id].getActiveness();
		}

		return totalActiveness;
	}

};

#endif // CPU_MONITOR_HPP
