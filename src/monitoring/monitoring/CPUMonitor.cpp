/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "CPUMonitor.hpp"


CPUStatistics *CPUMonitor::_cpuStatistics;
size_t CPUMonitor::_numCPUs;


void CPUMonitor::initialize()
{
	// Initialize the array of CPUStatistics
	std::vector<CPU *> const &cpus = CPUManager::getCPUListReference();
	_numCPUs = cpus.size();
	_cpuStatistics = new CPUStatistics[_numCPUs];
}

void CPUMonitor::shutdown()
{
	delete[] _cpuStatistics;
}

void CPUMonitor::displayStatistics(std::stringstream &stream)
{
	stream << std::left << std::fixed << std::setprecision(2) << "\n";
	stream << "+-----------------------------+\n";
	stream << "|       CPU STATISTICS        |\n";
	stream << "+-----------------------------+\n";
	stream << "|   CPU(id) - Activeness(%)   |\n";
	stream << "+-----------------------------+\n";

	// Iterate through all CPUs and print their ID and activeness
	for (unsigned short id = 0; id < _numCPUs; ++id) {
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
