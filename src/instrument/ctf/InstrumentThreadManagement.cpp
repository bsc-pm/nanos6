/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <unistd.h>
#include <sys/syscall.h>

#include "CTFTracepoints.hpp"
#include "InstrumentThreadManagement.hpp"
#include "ctfapi/CTFTypes.hpp"
#include "executors/threads/WorkerThread.hpp"


void Instrument::createdThread(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t const &computePlaceId)
{
	threadId = thread_id_t();
	ThreadLocalData &tld = getThreadLocalData();
	tld.isBusyWaiting = false;

	// At this point, we cannot use the per-cpu object. If this is a thread
	// created in the middle of the execution (i.e. outside nanos6
	// initialization) our parent might still be running and will own the
	// cpu. We will be able to make use of it after the worker thread has
	// performed the initial synchronization i.e. starting since
	// Instrument::threadSynchronizationCompleted().
}

void Instrument::precreatedExternalThread(/* OUT */ external_thread_id_t &threadId)
{
	// TODO: We should retrieve the thread id in a cleaner way
	ctf_thread_id_t tid = syscall(SYS_gettid);
	threadId.tid = tid;
	Instrument::tp_external_thread_create(tid);
}

void Instrument::threadSynchronizationCompleted(__attribute__((unused)) thread_id_t threadId)
{
	// TODO: We should use the thread_id_t to store and retrieve the TID
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != nullptr);
	Instrument::tp_thread_create(currentWorkerThread->getTid());
}

void Instrument::threadWillSuspend(
	__attribute__((unused)) thread_id_t threadId,
	__attribute__((unused)) compute_place_id_t cpu,
	bool afterSynchronization
) {
	if (afterSynchronization) {
		WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
		assert(currentWorkerThread != nullptr);
		Instrument::tp_thread_suspend(currentWorkerThread->getTid());
	}
}

void Instrument::threadHasResumed(
	__attribute__((unused)) thread_id_t threadId,
	__attribute__((unused)) compute_place_id_t cpu,
	bool afterSynchronization
) {
	if (afterSynchronization) {
		WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
		assert(currentWorkerThread != nullptr);
		Instrument::tp_thread_resume(currentWorkerThread->getTid());
	}
}

void Instrument::threadWillSuspend(external_thread_id_t threadId)
{
	Instrument::tp_external_thread_suspend(threadId.tid);
}

void Instrument::threadHasResumed(external_thread_id_t threadId)
{
	Instrument::tp_external_thread_resume(threadId.tid);
}

void Instrument::threadWillShutdown()
{
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != nullptr);
	Instrument::tp_thread_shutdown(currentWorkerThread->getTid());
}

void Instrument::threadWillShutdown(external_thread_id_t threadId)
{
	Instrument::tp_external_thread_shutdown(threadId.tid);
}
