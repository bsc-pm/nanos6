/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef WORKER_THREAD_BASE_HPP
#define WORKER_THREAD_BASE_HPP

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>

#include <InstrumentThreadManagement.hpp>
#include <InstrumentWorkerThread.hpp>

#include "executors/threads/CPU.hpp"
#include "hardware-counters/HardwareCounters.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "lowlevel/threads/KernelLevelThread.hpp"
#include "support/InstrumentedThread.hpp"

class WorkerThreadBase : protected KernelLevelThread, public InstrumentedThread {
protected:
	friend struct CPUThreadingModelData;

	//! The CPU on which this thread is running.
	CPU *_cpu;

	//! The CPU to which the thread transitions the next time it resumes. Atomic since this is changed by other threads.
	std::atomic<CPU *> _cpuToBeResumedOn;


	inline void markAsCurrentWorkerThread()
	{
		KernelLevelThread::setCurrentKernelLevelThread();
	}

	inline void synchronizeInitialization()
	{
		assert(_cpu != nullptr);
		bind(_cpu);

		// The thread suspends itself after initialization, since the "activator" is the one that will unblock it when needed
		Instrument::threadWillSuspend(_instrumentationId, _cpu->getInstrumentationId(), false);
		suspend();
		Instrument::threadSynchronizationCompleted(_instrumentationId);
		Instrument::threadHasResumed(_instrumentationId, _cpu->getInstrumentationId());
	}

	inline void start()
	{
		KernelLevelThread::start(_cpu->getPthreadAttr());
	}


public:
	inline WorkerThreadBase(CPU *cpu);
	virtual ~WorkerThreadBase()
	{
	}

	inline void suspend();

	//! \brief resume the thread on a given CPU
	//!
	//! \param[in] cpu the CPU on which to resume the thread
	//! \param[in] inInitializationOrShutdown true if it should not enforce assertions that are not valid during initialization and shutdown
	inline void resume(CPU *cpu, bool inInitializationOrShutdown);

	//! \brief suspend the currently running thread and replace it by another (if given)
	//!
	//! \param[in] replacement a thread that is currently suspended and that must take the place of the current thread or nullptr
	inline void switchTo(WorkerThreadBase *replacement);

	inline int getCpuId()
	{
		return _cpu->getSystemCPUId();
	}

	//! \brief get the hardware place currently assigned
	inline CPU *getComputePlace()
	{
		return _cpu;
	}

	//! \brief set the current hardware place
	//!
	//! Note: This function should only be used in very exceptional circumstances.
	inline void setComputePlace(CPU *cpu)
	{
		_cpu = cpu;
	}

	//! \brief returns the WorkerThread that runs the call
	static inline WorkerThreadBase *getCurrentWorkerThread()
	{
		return static_cast<WorkerThreadBase *> (getCurrentKernelLevelThread());
	}

	inline pid_t getTid()
	{
		return KernelLevelThread::getTid();
	}

};


WorkerThreadBase::WorkerThreadBase(CPU *cpu)
	: _cpu(cpu), _cpuToBeResumedOn(nullptr)
{
}


void WorkerThreadBase::suspend()
{
	Instrument::enterSuspend();
	KernelLevelThread::suspend();

	// Update the CPU since the thread may have migrated while blocked (or during pre-signaling)
	assert(_cpuToBeResumedOn != nullptr);
	_cpu = _cpuToBeResumedOn;

#ifndef NDEBUG
	_cpuToBeResumedOn = nullptr;
#endif
	Instrument::exitSuspend();
}


void WorkerThreadBase::resume(CPU *cpu, bool inInitializationOrShutdown)
{
	Instrument::enterResume();

	assert(cpu != nullptr);

	if (!inInitializationOrShutdown) {
		assert(KernelLevelThread::getCurrentKernelLevelThread() != this);
	}

	assert(_cpuToBeResumedOn == nullptr);
	_cpuToBeResumedOn.store(cpu, std::memory_order_release);
	if (_cpu != cpu) {
		Instrument::threadBindRemote(getInstrumentationId(), cpu->getInstrumentationId());
		bind(cpu);
	}

	if (!inInitializationOrShutdown) {
		assert(KernelLevelThread::getCurrentKernelLevelThread() != this);
	}

	// Resume it
	KernelLevelThread::resume();

	Instrument::exitResume();
}


void WorkerThreadBase::switchTo(WorkerThreadBase *replacement)
{
	Instrument::enterSwitchTo();
	assert(KernelLevelThread::getCurrentKernelLevelThread() == this);
	assert(replacement != this);

	CPU *cpu = _cpu;
	assert(cpu != nullptr);

	if (replacement != nullptr) {
		// Replace a thread by another
		replacement->resume(cpu, false);
	} else {
		// No replacement thread

		// NOTE1: In this case the CPUActivation class can end up resuming
		// a CPU before its running thread has had a chance to get blocked

		// NOTE2: The threadWillSuspend() instrumentation call cannot be
		// placed in this method when switching to nullptr because at
		// this point the thread's Nanos6 CPU object might no longer
		// belong to the thread. Therefore, it must be called before
		// this thread's CPU has been released.
	}

	Instrument::threadSuspending(_instrumentationId);
	suspend();
	// After resuming (if ever blocked), the thread continues here

	Instrument::threadHasResumed(_instrumentationId, _cpu->getInstrumentationId());
	Instrument::exitSwitchTo();
}



#endif // WORKER_THREAD_BASE_HPP
