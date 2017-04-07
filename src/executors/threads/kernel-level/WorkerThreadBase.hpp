#ifndef WORKER_THREAD_BASE_HPP
#define WORKER_THREAD_BASE_HPP

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <unistd.h>
#include <pthread.h>
#include <sys/types.h>


#include "executors/threads/CPU.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "lowlevel/KernelLevelThread.hpp"
#include "support/InstrumentedThread.hpp"

#include <InstrumentThreadManagement.hpp>


class WorkerThreadBase : protected KernelLevelThread, public InstrumentedThread {
protected:
	//! The CPU on which this thread is running.
	CPU *_cpu;
	
	//! The CPU to which the thread transitions the next time it resumes. Atomic since this is changed by other threads.
	std::atomic<CPU *> _cpuToBeResumedOn;
	
	inline void bind(CPU *cpu);
	
	inline void markAsCurrentWorkerThread()
	{
		KernelLevelThread::setCurrentKernelLevelThread();
	}
	
	
public:
	inline WorkerThreadBase(CPU * cpu);
	virtual ~WorkerThreadBase()
	{
	}
	
	inline void start();
	
	inline void suspend();
	
	//! \brief resume the thread on a given CPU
	//!
	//! \param[in] cpu the CPU on which to resume the thread
	//! \param[in] inInitializationOrShutdown true if it should not enforce assertions that are not valid during initialization and shutdown
	inline void resume(CPU *cpu, bool inInitializationOrShutdown);
	
	//! \brief migrate the currently running thread to a given CPU
	inline void migrate(CPU *cpu);
	
	
	//! \brief suspend the currently running thread and replace it by another (if given)
	//!
	//! \param[in] replacement a thread that is currently suspended and that must take the place of the current thread or nullptr
	inline void switchTo(WorkerThreadBase *replacement);
	
	
	inline int getCpuId()
	{
		return _cpu->_systemCPUId;
	}
	
	//! \brief get the hardware place currently assigned
	inline CPU *getComputePlace()
	{
		return _cpu;
	}
	
	
	//! \brief returns the WorkerThread that runs the call
	static inline WorkerThreadBase *getCurrentWorkerThread()
	{
		return static_cast<WorkerThreadBase *> (getCurrentKernelLevelThread());
	}
	
};


WorkerThreadBase::WorkerThreadBase(CPU* cpu)
	: _cpu(cpu), _cpuToBeResumedOn(nullptr)
{
}


void WorkerThreadBase::start()
{
	KernelLevelThread::start(&_cpu->_pthreadAttr);
}


void WorkerThreadBase::bind(CPU *cpu)
{
	int rc = sched_setaffinity(_tid, CPU_ALLOC_SIZE(cpu->_systemCPUId+1), &cpu->_cpuMask);
	FatalErrorHandler::handle(rc, " when changing affinity of pthread with thread id ", _tid, " to CPU ", cpu->_systemCPUId);
}


void WorkerThreadBase::suspend()
{
	KernelLevelThread::suspend();
}


void WorkerThreadBase::resume(CPU *cpu, bool inInitializationOrShutdown)
{
	assert(cpu != nullptr);
	
	if (!inInitializationOrShutdown) {
		assert(KernelLevelThread::getCurrentKernelLevelThread() != this);
	}
	
	assert(_cpuToBeResumedOn == nullptr);
	_cpuToBeResumedOn = cpu;
	if (_cpu != cpu) {
		bind(cpu);
	}
	
	if (!inInitializationOrShutdown) {
		assert(KernelLevelThread::getCurrentKernelLevelThread() != this);
	}
	
	// Resume it
	KernelLevelThread::resume();
}


void WorkerThreadBase::migrate(CPU *cpu)
{
	assert(cpu != nullptr);
	
	assert(KernelLevelThread::getCurrentKernelLevelThread() == this);
	assert(_cpu != cpu);
	
	assert(_cpuToBeResumedOn == nullptr);
	
	// Since it is the same thread the one that migrates itself, change the CPU directly
	_cpu = cpu;
	bind(cpu);
}


void WorkerThreadBase::switchTo(WorkerThreadBase *replacement)
{
	assert(KernelLevelThread::getCurrentKernelLevelThread() == this);
	assert(replacement != this);
	
	CPU *cpu = _cpu;
	assert(cpu != nullptr);
	
	if (replacement != nullptr) {
		// Replace a thread by another
		replacement->resume(cpu, false);
	} else {
		// No replacement thread
		
		// NOTE: In this case the CPUActivation class can end up resuming a CPU before its running thread has had a chance to get blocked
	}
	
	Instrument::threadWillSuspend(_instrumentationId, cpu->getInstrumentationId());
	
	suspend();
	// After resuming (if ever blocked), the thread continues here
	
	// Update the CPU since the thread may have migrated while blocked (or during pre-signaling)
	assert(_cpuToBeResumedOn != nullptr);
	_cpu = _cpuToBeResumedOn;
	
	Instrument::threadHasResumed(_instrumentationId, _cpu->getInstrumentationId());
	
#ifndef NDEBUG
	_cpuToBeResumedOn = nullptr;
#endif
}



#endif // WORKER_THREAD_BASE_HPP
