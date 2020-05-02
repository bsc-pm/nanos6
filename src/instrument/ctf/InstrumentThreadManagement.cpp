#define _GNU_SOURCE
#include <unistd.h>
#include <sys/syscall.h>

#include "executors/threads/WorkerThread.hpp"

#include "ctfapi/CTFTypes.hpp"
#include "Nanos6CTFEvents.hpp"
#include "InstrumentThreadManagement.hpp"

ctf_thread_id_t Instrument::gettid(void)
{
	return syscall(SYS_gettid);
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

void Instrument::createdThread(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t const &computePlaceId)
{
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != nullptr);
	Instrument::tp_thread_create(currentWorkerThread->getTid());
}
