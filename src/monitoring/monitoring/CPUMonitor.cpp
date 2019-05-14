/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "CPUMonitor.hpp"


CPUMonitor *CPUMonitor::_monitor;


void CPUMonitor::initialize()
{
	// Create the monitoring singleton
	if (_monitor == nullptr) {
		_monitor = new CPUMonitor();
	}
	
	// Initialize the array of CPUStatistics
	std::vector<CPU *> const &cpus = CPUManager::getCPUListReference();
	_monitor->_numCPUs = cpus.size();
	_monitor->_cpuStatistics = new CPUStatistics[_monitor->_numCPUs];
}

void CPUMonitor::shutdown()
{
	if (_monitor != nullptr) {
		delete[] _monitor->_cpuStatistics;
		
		delete _monitor;
	}
}

void CPUMonitor::displayStatistics(std::stringstream &stream)
{
	if (_monitor != nullptr) {
		stream << std::left << std::fixed << std::setprecision(2) << "\n";
		stream << "+-----------------------------+\n";
		stream << "|       CPU STATISTICS        |\n";
		stream << "+-----------------------------+\n";
		stream << "|   CPU(id) - Activeness(%)   |\n";
		stream << "+-----------------------------+\n";
		
		// Iterate through all CPUs and print their ID and activeness
		for (unsigned short id = 0; id < _monitor->_numCPUs; ++id) {
			std::string label = "CPU(" + std::to_string(id) + ")";
			float activeness = getActiveness(id);
			bool endOfColumn = (id % 2 || id == (_monitor->_numCPUs - 1));
			
			stream
				<< std::setw(8) << label << " - " << std::right
				<< std::setw(6) << (activeness * 100.00) << std::left << "%"
				<< (endOfColumn ? "\n" : " | ");
		}
		
		stream << "+-----------------------------+\n\n";
	}
}
