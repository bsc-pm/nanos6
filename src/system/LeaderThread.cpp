#include <cassert>
#include <ctime>
#include <iostream>

#include <time.h>

#include "LeaderThread.hpp"
#include "PollingAPI.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentLeaderThread.hpp>


LeaderThread *LeaderThread::_singleton;


void LeaderThread::initialize()
{
	_singleton = new LeaderThread();
}


void LeaderThread::shutdown()
{
	assert(_singleton != nullptr);
	bool expected = false;
	_singleton->_mustExit.compare_exchange_strong(expected, true);
	assert(!expected);
	
	void *dummy;
	int rc = pthread_join(_singleton->_pthread, &dummy);
	FatalErrorHandler::handle(rc, "Error joining leader thread");
	
	delete _singleton;
	_singleton = nullptr;
}


LeaderThread::LeaderThread()
	: _mustExit(false)
{
	start(nullptr);
}


void *LeaderThread::body()
{
	Instrument::ThreadInstrumentationContext instrumentationContext(Instrument::task_id_t(), Instrument::compute_place_id_t(), getInstrumentationId());
	
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


