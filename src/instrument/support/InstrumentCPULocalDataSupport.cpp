/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include <executors/threads/WorkerThread.hpp>
#include <system/LeaderThread.hpp>

#include "InstrumentCPULocalDataSupport.hpp"


Instrument::CPULocalData *Instrument::getCPULocalData()
{
	CPU *CPU;
	CPULocalData *cpuLocalData;
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();

	if (currentWorkerThread) {
		CPU = currentWorkerThread->getComputePlace();
		assert(CPU != nullptr);
		cpuLocalData = &CPU->getInstrumentationData();
	} else {
		if (LeaderThread::isLeaderThread()) {
			CPU = LeaderThread::getComputePlace();
			cpuLocalData = &CPU->getInstrumentationData();
		} else {
			cpuLocalData = virtualCPULocalData;
		}
	}

	assert(cpuLocalData != nullptr);
	return cpuLocalData;
}
