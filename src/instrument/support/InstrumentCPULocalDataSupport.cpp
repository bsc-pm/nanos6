/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include <executors/threads/WorkerThread.hpp>
#include <lowlevel/threads/ExternalThread.hpp>
#include <lowlevel/threads/ExternalThreadGroup.hpp>
#include "InstrumentCPULocalDataSupport.hpp"

Instrument::CPULocalData *Instrument::getCPULocalData()
{
	CPULocalData *cpuLocalData;
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();

	// TODO leader thread special case?
	// TODO what about external threads?
	if (currentWorkerThread) {
		CPU *CPU = currentWorkerThread->getComputePlace();
		assert(CPU != nullptr);
		cpuLocalData = &CPU->getInstrumentationData();
	} else {
		//ExternalThread *currentExternalThread = ExternalThread::getCurrentExternalThread();
		//if (currentExternalThread == nullptr) {
		//	//currentExternalThread = new ExternalThread("external");
		//	//assert(currentThread != nullptr);

		//	//currentExternalThread->initializeExternalThread();

		//	//// Register it in the group so that it will be deleted
		//	//// when shutting down Nanos6
		//	//ExternalThreadGroup::registerExternalThread(currentThread);
		//}

		// TODO use virtual CPU mechanism here
		cpuLocalData = virtualCPULocalData;
	}

	assert(cpuLocalData != nullptr);
	return cpuLocalData;
}
