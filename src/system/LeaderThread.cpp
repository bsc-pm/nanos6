/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2021 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <ctime>
#include <iostream>
#include <time.h>

#include "LeaderThread.hpp"
#include "support/config/ConfigVariable.hpp"

#include <InstrumentLeaderThread.hpp>
#include <InstrumentThreadManagement.hpp>


LeaderThread *LeaderThread::_singleton;


void LeaderThread::initialize(CPU *leaderThreadCPU)
{
	assert(leaderThreadCPU != nullptr);

	_singleton = new LeaderThread(leaderThreadCPU);
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

void LeaderThread::body()
{
	initializeHelperThread();
	Instrument::threadHasResumed(getInstrumentationId());

	while (!std::atomic_load_explicit(&_mustExit, std::memory_order_relaxed)) {
		// The delay is 1 millisecond
		struct timespec delay = {0, 1000000};

		// The loop repeats the call with the remaining time in the event that
		// the thread received a signal with a handler that has SA_RESTART set
		Instrument::threadWillSuspend(getInstrumentationId());
		while (nanosleep(&delay, &delay)) {
		}
		Instrument::threadHasResumed(getInstrumentationId());

		Instrument::leaderThreadSpin();
	}

	Instrument::threadWillShutdown(getInstrumentationId());

	return;
}
