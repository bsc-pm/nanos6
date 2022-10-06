/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef POSIX_KERNEL_LEVEL_THREAD_HPP
#define POSIX_KERNEL_LEVEL_THREAD_HPP

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cassert>

#include <pthread.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <MemoryAllocator.hpp>

#include "executors/threads/CPU.hpp"
#include "lowlevel/CompatSyscalls.hpp"
#include "lowlevel/ConditionVariable.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include <InstrumentPthread.hpp>


static void *kernel_level_thread_body_wrapper(void *parameter);


class KernelLevelThread {
protected:
	//! The underlying pthread
	pthread_t _pthread;
	pid_t _tid;
	pid_t _creatorTid; // Who created this thread

	//! This condition variable is used for suspending and resuming the thread
	ConditionVariable _suspensionConditionVariable;

	//! stack info to appropriate deallocate it
	size_t _stackSize;
	void *_stackPtr;

	//! Thread Local Storage variable to point back to the KernelLevelThread that is running the code
	static __thread KernelLevelThread *_currentKernelLevelThread;


	inline void exit()
	{
		pthread_exit(nullptr);
	}

	inline void setCurrentKernelLevelThread()
	{
		_currentKernelLevelThread = this;
	}

	friend void *kernel_level_thread_body_wrapper(void *parameter);


public:
	KernelLevelThread()
		: _stackSize(0), _stackPtr(nullptr)
	{
	}

	virtual ~KernelLevelThread()
	{
		if (_stackSize > 0) {
			assert(_stackPtr != nullptr);
			MemoryAllocator::free(_stackPtr, _stackSize);
		}
	}

	// WARNING: This should be only called by the thread initialization code
	inline void setTid(pid_t tid)
	{
		_tid = tid;
	}

	inline pid_t getTid()
	{
		return _tid;
	}

	inline pid_t getCreatorTid()
	{
		return _creatorTid;
	}

	inline void start(pthread_attr_t *pthreadAttr);

	inline void bind(CPU *cpu);

	//! \brief Suspend the thread
	inline void suspend()
	{
		Instrument::pthreadPause();
		_suspensionConditionVariable.wait();
		Instrument::pthreadResume();
	}

	//! \brief Resume the thread
	inline void resume()
	{
		Instrument::pthreadSignal(_tid);
		_suspensionConditionVariable.signal();
	}

	//! \brief Pauses the thread for the given time in nanoseconds
	inline int nsleep(const struct timespec *rqtp, struct timespec *rmtp);

	//! \brief Wait for the thread to finish and join it
	inline void join();

	//! \brief check if the thread will resume immediately when calling to suspend
	inline bool willResumeImmediately()
	{
		return _suspensionConditionVariable.isPresignaled();
	}

	//! \brief clear the pending resumption mark
	inline void abortResumption()
	{
		_suspensionConditionVariable.clearPresignal();
	}

	//! \brief code that the thread executes
	virtual void body() = 0;

	static inline KernelLevelThread *getCurrentKernelLevelThread()
	{
		return _currentKernelLevelThread;
	}

};


void *kernel_level_thread_body_wrapper(void *parameter)
{
	KernelLevelThread *thread = static_cast<KernelLevelThread *> (parameter);

	assert(thread != nullptr);
	thread->setTid(gettid());

	KernelLevelThread::_currentKernelLevelThread = thread;

	int cpu = -1; // The CPU is not set initially
	Instrument::pthreadBegin(thread->getCreatorTid(), cpu, parameter);

	thread->body();

	Instrument::pthreadEnd();

	return nullptr;
}


void KernelLevelThread::start(pthread_attr_t *pthreadAttr)
{
	void *stackptr;
	size_t stacksize;
	int rc;

	if (pthreadAttr != nullptr) {
		rc = pthread_attr_getstacksize(pthreadAttr, &stacksize);
		FatalErrorHandler::handle(rc, " when getting pthread's stacksize");

		stackptr = MemoryAllocator::alloc(stacksize);
		FatalErrorHandler::failIf(stackptr == nullptr, " when allocating pthread stack");
		_stackSize = stacksize;
		_stackPtr = stackptr;

		rc = pthread_attr_setstack(pthreadAttr, stackptr, stacksize);
		FatalErrorHandler::handle(rc, " when setting pthread's stack");
	}

	_creatorTid = getpid();
	int cpu = -1; // No affinity to one specific CPU
	Instrument::pthreadCreate(cpu, this);
	rc = pthread_create(&_pthread, pthreadAttr, &kernel_level_thread_body_wrapper, this);
	if (rc == EAGAIN) {
		FatalErrorHandler::failIf(true, " Insufficient resources when creating a pthread. This may happen due to:\n",
			"  (1) Having reached the system-imposed limit of threads\n",
			"  (2) The stack size limit is too large, try decreasing it with 'ulimit'");
	} else {
		FatalErrorHandler::handle(rc, " when creating a pthread");
	}
}

void KernelLevelThread::bind(CPU *cpu)
{
	assert(cpu != nullptr);
	Instrument::pthreadBind(_tid, cpu->getIndex());
	int rc = sched_setaffinity(_tid, CPU_ALLOC_SIZE(cpu->getSystemCPUId()+1), cpu->getCpuMask());
	FatalErrorHandler::handle(rc, " when changing affinity of pthread with thread id ", _tid, " to CPU ", cpu->getSystemCPUId());
}

int KernelLevelThread::nsleep(const struct timespec *rqtp, struct timespec *rmtp)
{
	Instrument::pthreadPause();
	int ret = nanosleep(rqtp, rmtp);
	Instrument::pthreadResume();

	return ret;
}

void KernelLevelThread::join()
{
	int rc = pthread_join(_pthread, nullptr);
	FatalErrorHandler::handle(rc, " during shutdown when joining pthread ", _pthread);
}



#endif // POSIX_KERNEL_LEVEL_THREAD_HPP
