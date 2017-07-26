/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef PAPI_HARDWARE_COUNTERS_THREAD_LOCAL_DATA_IMPLEMENTATION_HPP
#define PAPI_HARDWARE_COUNTERS_THREAD_LOCAL_DATA_IMPLEMENTATION_HPP



#include "PAPIHardwareCountersThreadLocalData.hpp"

#include "PAPIHardwareCounters.hpp"

#include "executors/threads/WorkerThread.hpp"
#include "executors/threads/WorkerThreadImplementation.hpp"


namespace HardwareCounters {
	namespace PAPI {
		inline HardwareCountersThreadLocalData &getCurrentThreadHardwareCounters()
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


#endif // PAPI_HARDWARE_COUNTERS_THREAD_LOCAL_DATA_IMPLEMENTATION_HPP
