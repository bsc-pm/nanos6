/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include "InstrumentCPULocalDataSupport.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "lowlevel/threads/ExternalThread.hpp"
#include "lowlevel/threads/ExternalThreadGroup.hpp"
#include "system/LeaderThread.hpp"


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
			// Ensure that the external thread object has been created
			ExternalThread *currentExternalThread = ExternalThread::getCurrentExternalThread();
			if (currentExternalThread == nullptr) {
				// Create a new ExternalThread structure for this
				// unknown external thread
				currentExternalThread = new ExternalThread("external");
				assert(currentExternalThread != nullptr);

				currentExternalThread->initializeExternalThread();

				// Register it in the group so that it will be deleted
				// when shutting down Nanos6
				ExternalThreadGroup::registerExternalThread(currentExternalThread);
			}
			cpuLocalData = nullptr;
		}
	}

	return cpuLocalData;
}
