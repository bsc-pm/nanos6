/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <config.h>
#include <sstream>

#include "CPUManagerInterface.hpp"
#include "system/RuntimeInfo.hpp"


std::vector<CPU *> CPUManagerInterface::_cpus;
cpu_set_t CPUManagerInterface::_cpuMask;
std::atomic<bool> CPUManagerInterface::_finishedCPUInitialization;
EnvironmentVariable<size_t> CPUManagerInterface::_taskforGroups("NANOS6_TASKFOR_GROUPS", 1);
EnvironmentVariable<bool> CPUManagerInterface::_taskforGroupsReportEnabled("NANOS6_TASKFOR_GROUPS_REPORT", 0);


namespace cpumanager_internals {
	static inline std::string maskToRegionList(boost::dynamic_bitset<> const &mask, size_t size)
	{
		std::ostringstream oss;

		int start = -1;
		int end = -1;
		bool first = true;
		for (size_t i = 0; i < size + 1; i++) {
			if ((i < size) && mask[i]) {
				if (start == -1) {
					start = i;
				}
				end = i;
			} else if (end >= 0) {
				if (first) {
					first = false;
				} else {
					oss << ",";
				}
				if (end == start) {
					oss << start;
				} else {
					oss << start << "-" << end;
				}
				start = -1;
				end = -1;
			}
		}

		return oss.str();
	}
}

void CPUManagerInterface::reportInformation(size_t numSystemCPUs, size_t numNUMANodes)
{
	boost::dynamic_bitset<> processCPUMask(numSystemCPUs);

	std::vector<boost::dynamic_bitset<>> NUMANodeSystemMask(numNUMANodes);
	for (size_t i = 0; i < numNUMANodes; ++i) {
		NUMANodeSystemMask[i].resize(numSystemCPUs);
	}

	for (CPU *cpu : _cpus) {
		assert(cpu != nullptr);

		if (cpu->isOwned()) {
			size_t systemCPUId = cpu->getSystemCPUId();
			processCPUMask[systemCPUId] = true;
			NUMANodeSystemMask[cpu->getNumaNodeId()][systemCPUId] = true;
		}
	}

	RuntimeInfo::addEntry(
		"initial_cpu_list",
		"Initial CPU List",
		cpumanager_internals::maskToRegionList(processCPUMask, numSystemCPUs)
	);
	for (size_t i = 0; i < numNUMANodes; ++i) {
		std::ostringstream oss, oss2;

		oss << "numa_node_" << i << "_cpu_list";
		oss2 << "NUMA Node " << i << " CPU List";
		std::string cpuRegionList = cpumanager_internals::maskToRegionList(NUMANodeSystemMask[i], numSystemCPUs);

		RuntimeInfo::addEntry(oss.str(), oss2.str(), cpuRegionList);
	}
}
