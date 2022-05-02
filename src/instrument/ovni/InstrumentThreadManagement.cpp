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
#include "OVNITrace.hpp"


void Instrument::createdThread(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t const &computePlaceId)
{
	OVNI::threadInit();
	OVNI::threadCreate(computePlaceId._id, 0);
}

void Instrument::precreatedExternalThread(/* OUT */__attribute__((unused)) external_thread_id_t &threadId)
{
	OVNI::threadInit();
	OVNI::threadExecute(-1, -1, 0);
	OVNI::threadAttach();

	// External threads are assumed to be "paused" when precreated
	OVNI::threadPause();
}

void Instrument::threadSynchronizationCompleted(__attribute__((unused)) thread_id_t threadId)
{
	// TODO: We should use the thread_id_t to store and retrieve the TID
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != nullptr);
	int cpu = currentWorkerThread->getCpuId();
	OVNI::threadExecute(cpu, 0, 0);

	// Mark as paused since it will be immediately resumed
	OVNI::threadPause();
}

void Instrument::threadWillSuspend(
	__attribute__((unused)) thread_id_t threadId,
	__attribute__((unused)) compute_place_id_t cpu,
	bool afterSynchronization
) {
	if (afterSynchronization) {
		OVNI::threadPause();
	}
}

void Instrument::threadHasResumed(
	__attribute__((unused)) thread_id_t threadId,
	__attribute__((unused)) compute_place_id_t cpu,
	bool afterSynchronization
) {
	if (afterSynchronization) {
		OVNI::threadResume();
		OVNI::affinitySet(cpu._id);
	}
}

void Instrument::threadWillSuspend(__attribute__((unused)) external_thread_id_t threadId)
{
	OVNI::threadPause();
}

void Instrument::threadHasResumed(__attribute__((unused)) external_thread_id_t threadId)
{
	OVNI::threadResume();
}

void Instrument::threadWillShutdown()
{
	OVNI::threadEnd();
}

void Instrument::threadWillShutdown(__attribute__((unused)) external_thread_id_t threadId)
{
	OVNI::threadDetach();
	OVNI::threadEnd();
}
