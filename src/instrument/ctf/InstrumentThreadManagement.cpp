/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <unistd.h>
#include <sys/syscall.h>

#include "executors/threads/WorkerThread.hpp"

#include "ctfapi/CTFTypes.hpp"
#include "CTFTracepoints.hpp"
#include "InstrumentThreadManagement.hpp"

#if defined(__GLIBC__) && (__GLIBC__ < 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ < 30))
static ctf_thread_id_t gettid(void)
{
	return syscall(SYS_gettid);
}
#endif

void Instrument::createdThread(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t const &computePlaceId)
{
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != nullptr);
	Instrument::tp_thread_create(currentWorkerThread->getTid());
}

void Instrument::precreatedExternalThread(/* OUT */ external_thread_id_t &threadId)
{
	ctf_thread_id_t tid = gettid();
	threadId = external_thread_id_t(tid);
	Instrument::tp_external_thread_create(tid);
}

void Instrument::threadWillSuspend(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t cpu)
{
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != nullptr);
	Instrument::tp_thread_suspend(currentWorkerThread->getTid());
}

void Instrument::threadHasResumed(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t cpu)
{
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != nullptr);
	Instrument::tp_thread_resume(currentWorkerThread->getTid());
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
