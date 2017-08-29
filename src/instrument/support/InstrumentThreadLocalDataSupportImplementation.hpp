/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_THREAD_LOCAL_DATA_SUPPORT_IMPLEMENTATION_HPP
#define INSTRUMENT_THREAD_LOCAL_DATA_SUPPORT_IMPLEMENTATION_HPP


#include "InstrumentThreadLocalDataSupport.hpp"

#include <lowlevel/threads/ExternalThread.hpp>
#include <executors/threads/WorkerThread.hpp>

#include <cassert>


inline Instrument::ExternalThreadLocalData &Instrument::getExternalThreadLocalData()
{
	ExternalThread *currentThread = ExternalThread::getCurrentExternalThread();
	assert(currentThread != nullptr);
	
	return currentThread->getInstrumentationData();
}


inline Instrument::ThreadLocalData &Instrument::getSentinelNonWorkerThreadLocalData()
{
	static thread_local ThreadLocalData nonWorkerThreadLocalData;
	return nonWorkerThreadLocalData;
}


inline Instrument::ThreadLocalData &Instrument::getThreadLocalData()
{
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	if (currentWorkerThread != nullptr) {
		return currentWorkerThread->getInstrumentationData();
	} else {
		return getSentinelNonWorkerThreadLocalData();
	}
}


#endif // INSTRUMENT_THREAD_LOCAL_DATA_SUPPORT_IMPLEMENTATION_HPP
