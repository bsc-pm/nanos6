/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include <executors/threads/WorkerThread.hpp>
#include <lowlevel/threads/ExternalThread.hpp>
#include <lowlevel/threads/ExternalThreadGroup.hpp>
#include <system/LeaderThread.hpp>

#include "InstrumentCPULocalDataSupport.hpp"


Instrument::CPULocalData *Instrument::getCPULocalData()
{
	CPULocalData *cpuLocalData;
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();

	if (currentWorkerThread) {
		CPU *CPU = currentWorkerThread->getComputePlace();
		assert(CPU != nullptr);
		cpuLocalData = &CPU->getInstrumentationData();
	} else {
		ExternalThread *currentExternalThread = ExternalThread::getCurrentExternalThread();
		if (currentExternalThread == LeaderThread::getLeaderThread()) {
			// TODO The leader thread will have its own lock-free virtual CPU
			//CPU *CPU = LeaderThread::getCPU();
			//cpuLocalData = &CPU->getInstrumentationData();
			cpuLocalData = leaderThreadCPULocalData;
		} else {
			cpuLocalData = virtualCPULocalData;
		}
	}

	assert(cpuLocalData != nullptr);
	return cpuLocalData;
}
