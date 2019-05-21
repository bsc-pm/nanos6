/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/


#include "InstrumentPAPIHardwareCounters.hpp"
#include "InstrumentPAPIHardwareCountersThreadLocalData.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "executors/threads/WorkerThreadImplementation.hpp"


namespace InstrumentHardwareCounters {
	namespace PAPI {
		HardwareCountersThreadLocalData &getCurrentThreadHardwareCounters()
		{
			WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
			if (currentWorkerThread != nullptr) {
				return currentWorkerThread->getHardwareCounters();
			} else {
				static thread_local HardwareCountersThreadLocalData nonWorkerHardwareCounters;
				return nonWorkerHardwareCounters;
			}
		}
	}
}

