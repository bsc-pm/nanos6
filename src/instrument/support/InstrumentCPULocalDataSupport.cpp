/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <executors/threads/WorkerThread.hpp>
#include "InstrumentCPULocalDataSupport.hpp"

Instrument::CPULocalData &Instrument::getCPULocalData()
{
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	CPU *CPU;

	// TODO leader thread special case?
	// TODO what about external threads?
	//if (currentWorkerThread != nullptr) {
	//	CPU = currentWorkerThread->getComputePlace();
	//	//TODO what is currentCPU is null?
	//} else {
	//	// TODO no idea what to do here
	//}

	assert(correntWorkerThread != nullptr);
	CPU = currentWorkerThread->getComputePlace();
	assert(CPU != nullptr);

	return CPU->getInstrumentationData();
}
