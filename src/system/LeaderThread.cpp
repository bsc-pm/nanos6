#include <ctime>
#include <iostream>

#include <time.h>

#include "LeaderThread.hpp"

#include <InstrumentLeaderThread.hpp>


std::atomic<bool> LeaderThread::_mustExit(false);


void LeaderThread::maintenanceLoop()
{
	while (!std::atomic_load_explicit(&_mustExit, std::memory_order_relaxed)) {
		struct timespec delay = { 0, 1000000}; // 1000 Hz
		
		// The loop repeats the call with the remaining time in the event that the thread received a signal with a handler that has SA_RESTART set
		while (nanosleep(&delay, &delay)) {
		}
		
		// check somenting, like changes on the process_mask, or if there is any
		// pending work
		Instrument::leaderThreadSpin();
	}
}


void LeaderThread::notifyMainExit()
{
	std::atomic_store_explicit(&_mustExit, true, std::memory_order_release);
}

