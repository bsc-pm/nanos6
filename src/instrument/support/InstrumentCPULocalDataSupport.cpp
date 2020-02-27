/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include "InstrumentCPULocalDataSupport.hpp"

Instrument::CPULocalData &Instrument::getCPULocalData()
{
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	Instrumentation::CPULocalData *currentCPULocalData;

	// TODO leader thread special case?
	// TODO what about external threads?
	if (currentWorkerThread != nullptr) {
		CPU *CPU = currentThread->getComputePlace();
		//TODO what is currentCPU is null?
		currentCPULocalData = CPU->getInstrumentationData();
	} else {
		// TODO no idea what to do here
	}

	return currentCPULocalData;
}
