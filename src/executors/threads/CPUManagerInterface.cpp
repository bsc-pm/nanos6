/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <config.h>
#include <sstream>

#include "CPUManagerInterface.hpp"
#include "system/RuntimeInfo.hpp"


std::vector<CPU *> CPUManagerInterface::_cpus;
std::vector<size_t> CPUManagerInterface::_systemToVirtualCPUId;
cpu_set_t CPUManagerInterface::_cpuMask;
std::atomic<bool> CPUManagerInterface::_finishedCPUInitialization;
ConfigVariable<size_t> CPUManagerInterface::_taskforGroups("taskfor.groups");
ConfigVariable<bool> CPUManagerInterface::_taskforGroupsReportEnabled("taskfor.report");
CPUManagerPolicyInterface *CPUManagerInterface::_cpuManagerPolicy;
ConfigVariable<std::string> CPUManagerInterface::_policyChosen("cpumanager.policy");
CPUManagerPolicy CPUManagerInterface::_policyId;
size_t CPUManagerInterface::_firstCPUId;
CPU *CPUManagerInterface::_leaderThreadCPU;


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

void CPUManagerInterface::refineTaskforGroups(size_t numCPUs, size_t numNUMANodes)
{
	// Whether the taskfor group envvar already has a value
	bool taskforGroupsSetByUser = _taskforGroups.isPresent();

	// Final warning message (only one)
	bool mustEmitWarning = false;
	std::ostringstream warningMessage;

	// The default value is the closest to 1 taskfor group per NUMA node
	if (!taskforGroupsSetByUser) {
		size_t closestGroups = numNUMANodes;
		if (numCPUs % numNUMANodes != 0) {
			closestGroups = getClosestGroupNumber(numCPUs, numNUMANodes);
			assert(numCPUs % closestGroups == 0);
		}
		_taskforGroups.setValue(closestGroups);
	} else {
		if (numCPUs < _taskforGroups) {
			warningMessage
				<< "More groups requested than available CPUs. "
				<< "Using " << numCPUs << " groups of 1 CPU each instead";

			_taskforGroups.setValue(numCPUs);
			mustEmitWarning = true;
		} else if (_taskforGroups == 0 || numCPUs % _taskforGroups != 0) {
			size_t closestGroups = getClosestGroupNumber(numCPUs, _taskforGroups);
			assert(numCPUs % closestGroups == 0);

			size_t cpusPerGroup = numCPUs / closestGroups;
			warningMessage
				<< _taskforGroups << " groups requested. "
				<< "The number of CPUs is not divisible by the number of groups. "
				<< "Using " << closestGroups << " groups of " << cpusPerGroup
				<< " CPUs each instead";

			_taskforGroups.setValue(closestGroups);
			mustEmitWarning = true;
		}
	}

	if (mustEmitWarning) {
		FatalErrorHandler::warnIf(true, warningMessage.str());
	}

	assert((_taskforGroups <= numCPUs) && (numCPUs % _taskforGroups == 0));
}

void CPUManagerInterface::reportTaskforGroupsInfo()
{
	std::ostringstream report;
	size_t numTaskforGroups = getNumTaskforGroups();
	size_t numCPUsPerTaskforGroup = getNumCPUsPerTaskforGroup();
	report << "There are " << numTaskforGroups << " taskfor groups with "
		<< numCPUsPerTaskforGroup << " CPUs each." << std::endl;

	std::vector<std::vector<size_t>> cpusPerGroup(numTaskforGroups);
	for (size_t cpu = 0; cpu < _cpus.size(); cpu++) {
		assert(_cpus[cpu]->getIndex() == (int) cpu);

		size_t groupId = _cpus[cpu]->getGroupId();
		cpusPerGroup[groupId].push_back(cpu);
	}

	for (size_t group = 0; group < numTaskforGroups; group++) {
		report << "Group " << group << " contains the following CPUs:" << std::endl;
		report << "{";

		size_t groupSize = cpusPerGroup[group].size();
		for (size_t i = 0; i < groupSize; i++) {
			size_t cpuId = cpusPerGroup[group][i];
			size_t systemCPUId = _cpus[cpuId]->getSystemCPUId();
			assert(_cpus[cpuId]->getGroupId() == group);

			report << systemCPUId;
			if (i < (groupSize - 1)) {
				report << ",";
			}
		}
		assert(group == 0 || groupSize == cpusPerGroup[group - 1].size());

		report << "}" << std::endl;
	}
}

