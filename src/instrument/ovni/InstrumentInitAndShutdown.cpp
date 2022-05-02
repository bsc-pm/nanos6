/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "InstrumentCPULocalData.hpp"
#include "InstrumentInitAndShutdown.hpp"
#include "executors/threads/CPUManager.hpp"
#include "hardware-counters/HardwareCounters.hpp"
#include "memory/numa/NUMAManager.hpp"
#include "tasks/TaskInfo.hpp"
#include "tasks/TasktypeData.hpp"
#include "OVNITrace.hpp"


void Instrument::initialize()
{
	OVNI::procInit();

	CPU *cpu;
	std::vector<CPU *> cpus = CPUManager::getCPUListReference();
	size_t totalCPUs = cpus.size();

	for (size_t i = 0; i < totalCPUs; ++i)
		OVNI::cpuId(i, cpus[i]->getSystemCPUId());
}

void Instrument::shutdown()
{
	OVNI::procFini();
}

void Instrument::preinitFinished()
{
	// emit an event per each registered task type with its label and source
	TaskInfo::processAllTasktypes(
		[&](const std::string &tasktypeLabel, const std::string &tasktypeSource, TasktypeData &tasktypeData) {
			TasktypeInstrument &instrumentId = tasktypeData.getInstrumentationId();
			uint32_t tasktypeId = instrumentId.autoAssingId();
			OVNI::typeCreate(tasktypeId, tasktypeLabel.c_str());
		}
	);
}

int64_t Instrument::getInstrumentStartTime()
{
	// TODO
	return 0;
}
