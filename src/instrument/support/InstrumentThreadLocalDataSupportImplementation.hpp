#ifndef INSTRUMENT_THREAD_LOCAL_DATA_SUPPORT_IMPLEMENTATION_HPP
#define INSTRUMENT_THREAD_LOCAL_DATA_SUPPORT_IMPLEMENTATION_HPP


#include "InstrumentThreadLocalDataSupport.hpp"

#include <executors/threads/WorkerThread.hpp>


inline Instrument::ThreadLocalData &Instrument::getThreadLocalData()
{
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	if (currentWorkerThread != nullptr) {
		return currentWorkerThread->getInstrumentationData();
	} else {
		static thread_local ThreadLocalData nonWorkerThreadLocalData;
		return nonWorkerThreadLocalData;
	}
}


#endif // INSTRUMENT_THREAD_LOCAL_DATA_SUPPORT_IMPLEMENTATION_HPP
