/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <unistd.h>
#include <sys/syscall.h>

#include "executors/threads/WorkerThread.hpp"
#include "InstrumentThreadManagement.hpp"
#include "OvniTrace.hpp"


void Instrument::createdThread(__attribute__((unused)) thread_id_t threadId,
		__attribute__((unused)) compute_place_id_t const &computePlaceId)
{
}

void Instrument::precreatedExternalThread(/* OUT */__attribute__((unused)) external_thread_id_t &threadId)
{
}

void Instrument::threadSynchronizationCompleted(__attribute__((unused)) thread_id_t threadId)
{
}

void Instrument::threadWillSuspend(
	__attribute__((unused)) thread_id_t threadId,
	__attribute__((unused)) compute_place_id_t cpu,
	bool afterSynchronization
) {
	if (afterSynchronization) {
		Ovni::threadCool();
	}
}

void Instrument::threadSuspending(__attribute__((unused)) thread_id_t threadId)
{
}

void Instrument::threadBindRemote(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t cpu)
{
}

void Instrument::threadHasResumed(
	__attribute__((unused)) thread_id_t threadId,
	__attribute__((unused)) compute_place_id_t cpu,
	__attribute__((unused)) bool afterSynchronization
) {
}

void Instrument::threadWillSuspend(__attribute__((unused)) external_thread_id_t threadId)
{
}

void Instrument::threadHasResumed(__attribute__((unused)) external_thread_id_t threadId)
{
}

void Instrument::threadWillShutdown()
{
}

void Instrument::threadWillShutdown(__attribute__((unused)) external_thread_id_t threadId)
{
}
