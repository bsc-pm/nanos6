/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#include "InstrumentInitAndShutdown.hpp"
#include "executors/threads/CPUManager.hpp"
#include "tasks/TaskInfoManager.hpp"
#include "OvniTrace.hpp"


void Instrument::initialize()
{
	// Too late, we initialize the ovni process and thread earlier at
	// mainThreadBegin.
}

void Instrument::addCPUs()
{
	std::vector<CPU *> const &cpus = CPUManager::getCPUListReference();
	size_t totalCPUs = cpus.size();

	for (size_t i = 0; i < totalCPUs; ++i)
		Ovni::addCPU(i, cpus[i]->getSystemCPUId());
}

void Instrument::shutdown()
{
	// Too early, shutdown done in mainThreadEnd.
}

void Instrument::preinitFinished()
{
	// Emit an event per each registered task info with its label
	TaskInfoManager::processAllTaskInfos(
		[&](const nanos6_task_info_t *, TaskInfoData &taskInfoData) {
			TasktypeInstrument &instrumentId = taskInfoData.getInstrumentationId();
			uint32_t tasktypeId = instrumentId.assignNewId();
			Ovni::typeCreate(tasktypeId, taskInfoData.getTaskTypeLabel().c_str());
		}
	);
}

int64_t Instrument::getInstrumentStartTime()
{
	// No need, the offset is computed by another ovni tool
	return 0;
}
