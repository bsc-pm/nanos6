/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/


#include <cassert>

#include "InstrumentThreadLocalDataSupport.hpp"

#include <executors/threads/WorkerThread.hpp>
#include <lowlevel/threads/ExternalThread.hpp>
#include <lowlevel/threads/ExternalThreadGroup.hpp>


Instrument::ExternalThreadLocalData &Instrument::getExternalThreadLocalData()
{
	ExternalThread *currentThread = ExternalThread::getCurrentExternalThread();
	if (currentThread == nullptr) {
		// Create a new ExternalThread structure for this
		// unknown external thread
		currentThread = new ExternalThread("external");
		assert(currentThread != nullptr);
		
		currentThread->initializeExternalThread();
		
		// Register it in the group so that it will be deleted
		// when shutting down Nanos6
		ExternalThreadGroup::registerExternalThread(currentThread);
	}
	assert(currentThread != nullptr);
	
	return currentThread->getInstrumentationData();
}


Instrument::ThreadLocalData &Instrument::getThreadLocalData()
{
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	if (currentWorkerThread != nullptr) {
		return currentWorkerThread->getInstrumentationData();
	} else {
		return getSentinelNonWorkerThreadLocalData();
	}
}

