/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "ThreadHardwareCountersMonitor.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


ThreadHardwareCountersMonitor *ThreadHardwareCountersMonitor::_monitor;


void ThreadHardwareCountersMonitor::initializeThread(
	ThreadHardwareCounters *threadCounters,
	pqos_mon_event monitoredEvents
) {
	assert(threadCounters != nullptr);
	
	// Allocate PQoS event structures
	pqos_mon_data *threadData = (pqos_mon_data *) malloc(sizeof(pqos_mon_data));
	FatalErrorHandler::failIf(threadData == nullptr, "Could not allocate memory for a thread's hardware counter structures");
	
	// Link the structures to the current thread
	threadCounters->setData(threadData);
	threadCounters->setTid(WorkerThread::getCurrentWorkerThread()->getTid());
	
	// Begin PQoS monitoring for the current thread
	int ret = pqos_mon_start_pid(
		threadCounters->getTid(),
		monitoredEvents,
		NULL,
		threadCounters->getData()
	);
	FatalErrorHandler::failIf(ret != PQOS_RETVAL_OK, "Error '", ret, "' when initializing hardware counter monitoring (PQoS) for a thread");
}

void ThreadHardwareCountersMonitor::shutdownThread(ThreadHardwareCounters *threadCounters)
{
	assert(threadCounters != nullptr);
	
	// Finish PQoS monitoring for the current thread
	int ret = pqos_mon_stop(threadCounters->getData());
	FatalErrorHandler::failIf(ret != PQOS_RETVAL_OK, "Error '", ret, "' when stopping hardware counter monitoring (PQoS) for a thread");
}
