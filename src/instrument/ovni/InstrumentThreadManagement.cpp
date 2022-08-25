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

	WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
	if (currentWorker != nullptr)
		currentWorker->setInstrumentationId(currentWorker->getTid());
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
		Ovni::threadCool();
	}
}

void Instrument::threadSuspending(__attribute__((unused)) thread_id_t threadId)
{
	Ovni::threadPause();
}

void Instrument::threadBindRemote(thread_id_t threadId,	compute_place_id_t cpu)
{
	Ovni::affinityRemote(cpu._id, threadId);
}

void Instrument::threadHasResumed(
	__attribute__((unused)) thread_id_t threadId,
	__attribute__((unused)) compute_place_id_t cpu,
	bool afterSynchronization
) {
	if (afterSynchronization) {
		Ovni::threadResume();
	}
}

void Instrument::threadWillSuspend(__attribute__((unused)) external_thread_id_t threadId)
{
	// For the case of external threads, as they do not have to wake up anyone else, they don't have
	// a cooling state
	Ovni::threadPause();
}

void Instrument::threadHasResumed(__attribute__((unused)) external_thread_id_t threadId)
{
	Ovni::threadResume();
}

void Instrument::threadWillShutdown()
{
	ThreadLocalData &tld = getThreadLocalData();
	if (tld._hungry) {
		tld._hungry = false;
		Ovni::schedFill();
	}

	Ovni::threadEnd();
}

void Instrument::threadWillShutdown(__attribute__((unused)) external_thread_id_t threadId)
{
	Ovni::threadDetach();
	Ovni::threadEnd();
}
