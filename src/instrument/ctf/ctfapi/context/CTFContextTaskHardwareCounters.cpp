/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <vector>

#include "tasks/Task.hpp"
#include "hardware-counters/HardwareCounters.hpp"
#include "hardware-counters/TaskHardwareCounters.hpp"
#include "executors/threads/WorkerThread.hpp"

#include "../CTFAPI.hpp"
#include "CTFContextTaskHardwareCounters.hpp"

CTFAPI::CTFContextTaskHardwareCounters::CTFContextTaskHardwareCounters() : CTFContext()
{
	const std::vector<HWCounters::counters_t> &enabledCounters = HardwareCounters::getEnabledCounters();

	eventMetadata.append("\t\tstruct hwc hwc;\n");
	dataStructuresMetadata.append("struct hwc {\n");
	for (auto it = enabledCounters.begin(); it != enabledCounters.end(); ++it) {
		dataStructuresMetadata.append(
			std::string("\tuint64_t ") +
			HWCounters::counterDescriptions[*it] +
			";\n"
		);
	}
	dataStructuresMetadata.append("};\n\n");
	size = sizeof(uint64_t) * enabledCounters.size();
}

void CTFAPI::CTFContextTaskHardwareCounters::writeContext(void **buf)
{
	const std::vector<HWCounters::counters_t> &enabledCounters = HardwareCounters::getEnabledCounters();
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	Task *task = currentWorkerThread->getTask();
	TaskHardwareCounters &taskCounters = task->getHardwareCounters();

	for (auto it = enabledCounters.begin(); it != enabledCounters.end(); ++it) {
		uint64_t val = (uint64_t) taskCounters.getDelta(*it);
		tp_write_args(buf, val);
	}
}
