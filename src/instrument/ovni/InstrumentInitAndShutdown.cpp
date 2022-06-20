/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#include "InstrumentInitAndShutdown.hpp"
#include "tasks/TaskInfo.hpp"
#include "tasks/TasktypeData.hpp"
#include "OvniTrace.hpp"


void Instrument::initialize()
{
	Ovni::initialize();

	std::vector<CPU *> const &cpus = CPUManager::getCPUListReference();
	size_t totalCPUs = cpus.size();

	for (size_t i = 0; i < totalCPUs; ++i)
		Ovni::addCPU(i, cpus[i]->getSystemCPUId());
}

void Instrument::shutdown()
{
	Ovni::finalize();
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
	// TODO
	return 0;
}
