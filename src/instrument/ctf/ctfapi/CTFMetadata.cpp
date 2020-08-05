/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/


#include "CTFMetadata.hpp"
#include "CTFTypes.hpp"
#include "executors/threads/CPUManager.hpp"


std::string CTFAPI::CTFMetadata::_cpuList;

void CTFAPI::CTFMetadata::collectCommonInformation()
{
	std::vector<CPU *> cpus = CPUManager::getCPUListReference();
	ctf_cpu_id_t totalCPUs = (ctf_cpu_id_t) cpus.size();

	ctf_cpu_id_t cpuId = cpus[0]->getSystemCPUId();
	_cpuList = std::to_string(cpuId);
	for (ctf_cpu_id_t i = 1; i < totalCPUs; i++) {
		cpuId = cpus[i]->getSystemCPUId();
		_cpuList += "," + std::to_string(cpuId);
	}
}
