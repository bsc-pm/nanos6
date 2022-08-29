/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#include "InstrumentInitAndShutdown.hpp"
#include "executors/threads/CPUManager.hpp"
#include "tasks/TaskInfo.hpp"
#include "tasks/TasktypeData.hpp"
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
	// Emit an event per each registered task type with its label and source
	TaskInfo::processAllTasktypes(
		[&](const std::string &tasktypeLabel, const std::string &, TasktypeData &tasktypeData) {
			TasktypeInstrument &instrumentId = tasktypeData.getInstrumentationId();
			uint32_t tasktypeId = instrumentId.assignNewId();
			Ovni::typeCreate(tasktypeId, tasktypeLabel.c_str());
		}
	);
}

int64_t Instrument::getInstrumentStartTime()
{
	// No need, the offset is computed by another ovni tool
	return 0;
}
