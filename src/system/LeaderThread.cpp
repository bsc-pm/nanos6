#include <cassert>
#include <ctime>
#include <iostream>

#include <time.h>

#include "LeaderThread.hpp"
#include "PollingAPI.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentLeaderThread.hpp>
#include <InstrumentThreadManagement.hpp>


LeaderThread *LeaderThread::_singleton;


void LeaderThread::initialize()
{
	_singleton = new LeaderThread();
	_singleton->start(nullptr);
}


void LeaderThread::shutdown()
{
	assert(_singleton != nullptr);
	bool expected = false;
	_singleton->_mustExit.compare_exchange_strong(expected, true);
	assert(!expected);
	
	_singleton->join();
	
	delete _singleton;
	_singleton = nullptr;
}


LeaderThread::LeaderThread()
	: _mustExit(false)
{
}


void *LeaderThread::body()
{
	Instrument::task_id_t instrumentationTaskId;
	Instrument::compute_place_id_t instrumentationComputePlaceId;
	Instrument::thread_id_t instrumentationThreadId;
	
	Instrument::ThreadInstrumentationContext instrumentationContext(
		instrumentationTaskId, instrumentationComputePlaceId, instrumentationThreadId
	);
	
	while (!std::atomic_load_explicit(&_mustExit, std::memory_order_relaxed)) {
		struct timespec delay = { 0, 1000000}; // 1000 Hz
		
		// The loop repeats the call with the remaining time in the event that the thread received a signal with a handler that has SA_RESTART set
		while (nanosleep(&delay, &delay)) {
		}
		
		PollingAPI::handleServices();
		
		Instrument::leaderThreadSpin(instrumentationContext.get());
	}
	
	return nullptr;
}


