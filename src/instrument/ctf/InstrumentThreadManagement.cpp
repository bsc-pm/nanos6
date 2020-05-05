#define _GNU_SOURCE
#include <unistd.h>
#include <sys/syscall.h>

#include "executors/threads/WorkerThread.hpp"

#include "ctfapi/CTFTypes.hpp"
#include "Nanos6CTFEvents.hpp"
#include "InstrumentThreadManagement.hpp"

static ctf_thread_id_t gettid(void)
{
	return syscall(SYS_gettid);
}

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
	Instrument::tp_thread_suspend(threadId.tid);
}
void Instrument::threadHasResumed(external_thread_id_t threadId)
{
	Instrument::tp_thread_resume(threadId.tid);
}
