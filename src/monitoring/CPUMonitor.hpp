/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_MONITOR_HPP
#define CPU_MONITOR_HPP

#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "CPUStatistics.hpp"
#include "executors/threads/CPUManager.hpp"


class CPUMonitor {

private:

	//! An array of CPU statistics, one per CPU
	CPUStatistics *_cpuStatistics;

	//! The amount of CPUs of the system
	size_t _numCPUs;

public:

	inline CPUMonitor()
	{
		// Make sure the CPUManager is already preinitialized before this
		assert(CPUManager::isPreinitialized());

		_numCPUs = CPUManager::getTotalCPUs();
		_cpuStatistics = new CPUStatistics[_numCPUs];
		assert(_cpuStatistics != nullptr);
	}

	inline ~CPUMonitor()
	{
		assert(_cpuStatistics != nullptr);

		delete[] _cpuStatistics;
		_cpuStatistics = nullptr;
	}

	//! \brief Signal that a CPU just became active
	//!
	//! \param[in] virtualCPUId The identifier of the CPU
	inline void cpuBecomesActive(int virtualCPUId)
	{
		_cpuStatistics[virtualCPUId].cpuBecomesActive();
	}

	//! \brief Signal that a CPU just became idle
	//!
	//! \param[in] virtualCPUId The identifier of the CPU
	inline void cpuBecomesIdle(int virtualCPUId)
	{
		_cpuStatistics[virtualCPUId].cpuBecomesIdle();
	}

	//! \brief Retreive the activeness of a CPU
	//!
	//! \param[in] virtualCPUId The identifier of the CPU
	inline float getActiveness(int virtualCPUId)
	{
		return _cpuStatistics[virtualCPUId].getActiveness();
	}

	//! \brief Get the total amount of activeness of all CPUs
	inline float getTotalActiveness()
	{
		float totalActiveness = 0.0;
		for (size_t id = 0; id < _numCPUs; ++id) {
			totalActiveness += _cpuStatistics[id].getActiveness();
		}

		return totalActiveness;
	}

	//! \brief Return the number of CPUs in the system
	inline size_t getNumCPUs() const
	{
		return _numCPUs;
	}

	//! \brief Display CPU statistics
	//!
	//! \param[out] stream The output stream
	inline void displayStatistics(std::stringstream &stream)
	{
		stream << std::left << std::fixed << std::setprecision(2) << "\n";
		stream << "+-----------------------------+\n";
		stream << "|       CPU STATISTICS        |\n";
		stream << "+-----------------------------+\n";
		stream << "|   CPU(id) - Activeness(%)   |\n";
		stream << "+-----------------------------+\n";

		// Iterate through all CPUs and print their ID and activeness
		for (size_t id = 0; id < _numCPUs; ++id) {
			std::string label = "CPU(" + std::to_string(id) + ")";
			float activeness = getActiveness(id);
			bool endOfColumn = (id % 2 || id == (_numCPUs - 1));

			stream
				<< std::setw(8) << label << " - " << std::right
				<< std::setw(6) << (activeness * 100.00) << std::left << "%"
				<< (endOfColumn ? "\n" : " | ");
		}

		stream << "+-----------------------------+\n\n";
	}

};

#endif // CPU_MONITOR_HPP
