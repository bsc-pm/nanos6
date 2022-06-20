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


void Instrument::createdThread(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t const &computePlaceId)
{
	Ovni::threadInit();
	Ovni::threadCreate(computePlaceId._id, 0);
}

void Instrument::precreatedExternalThread(/* OUT */__attribute__((unused)) external_thread_id_t &threadId)
{
	Ovni::threadInit();
	Ovni::threadExecute(-1, -1, 0);
	Ovni::threadAttach();

	// External threads are assumed to be "paused" when precreated
	Ovni::threadPause();
}

void Instrument::threadSynchronizationCompleted(__attribute__((unused)) thread_id_t threadId)
{
	// TODO: We should use the thread_id_t to store and retrieve the TID
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != nullptr);
	int cpu = currentWorkerThread->getComputePlace()->getIndex();
	Ovni::threadExecute(cpu, 0, 0);

	// Mark as paused since it will be immediately resumed
	Ovni::threadPause();
}

void Instrument::threadWillSuspend(
	__attribute__((unused)) thread_id_t threadId,
	__attribute__((unused)) compute_place_id_t cpu,
	bool afterSynchronization
) {
	if (afterSynchronization) {
		Ovni::threadPause();
	}
}

void Instrument::threadHasResumed(
	__attribute__((unused)) thread_id_t threadId,
	__attribute__((unused)) compute_place_id_t cpu,
	bool afterSynchronization
) {
	if (afterSynchronization) {
		Ovni::threadResume();
		Ovni::affinitySet(cpu._id);
	}
}

void Instrument::threadWillSuspend(__attribute__((unused)) external_thread_id_t threadId)
{
	Ovni::threadPause();
}

void Instrument::threadHasResumed(__attribute__((unused)) external_thread_id_t threadId)
{
	Ovni::threadResume();
}

void Instrument::threadWillShutdown()
{
	Ovni::threadEnd();
}

void Instrument::threadWillShutdown(__attribute__((unused)) external_thread_id_t threadId)
{
	Ovni::threadDetach();
	Ovni::threadEnd();
}
