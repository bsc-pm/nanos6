/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <vector>

#include "hardware-counters/HardwareCounters.hpp"
#include "hardware-counters/CPUHardwareCounters.hpp"
#include "executors/threads/WorkerThread.hpp"

#include "instrument/ctf/ctfapi/CTFAPI.hpp"
#include "CTFContextCPUHardwareCounters.hpp"

CTFAPI::CTFContextCPUHardwareCounters::CTFContextCPUHardwareCounters(ctf_stream_id_t streamMask) : CTFEventContext(streamMask)
{
	const std::vector<HWCounters::counters_t> &enabledCounters = HardwareCounters::getEnabledCounters();

	// the definition of the CTF "struct hwc" is under
	// CTFContextTaskHardwareCounters.

	// TODO it is ugly to have the definition in only one
	// side. Adapt the design to avoid having two contexes
	// that share the same "CTF struct"

	eventMetadata.append("\t\tstruct hwc hwc;\n");
	size = sizeof(uint64_t) * enabledCounters.size();
}


void CTFAPI::CTFContextCPUHardwareCounters::writeContext(void **buf, ctf_stream_id_t streamId)
{
	if (!(streamId & _streamMask))
		return;

	const std::vector<HWCounters::counters_t> &enabledCounters = HardwareCounters::getEnabledCounters();
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();

	// HardwareCounters are only written by worker threads
	assert(currentWorkerThread != nullptr);

	CPU *cpu = currentWorkerThread->getComputePlace();
	CPUHardwareCounters &CPUCounters = cpu->getHardwareCounters();

	for (auto it = enabledCounters.begin(); it != enabledCounters.end(); ++it) {
		uint64_t val = (uint64_t) CPUCounters.getDelta(*it);
		tp_write_args(buf, val);
	}
}
