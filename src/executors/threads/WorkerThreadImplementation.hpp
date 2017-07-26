/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef WORKER_THREAD_IMPLEMENTATION_HPP
#define WORKER_THREAD_IMPLEMENTATION_HPP


#include "CPU.hpp"
#include "DependencyDomain.hpp"
#include "WorkerThread.hpp"
#include "WorkerThreadBase.hpp"
#include "performance/HardwareCountersThreadLocalData.hpp"

#include <cassert>
#include <typeinfo>


inline WorkerThread::WorkerThread(CPU * cpu)
	: WorkerThreadBase(cpu), _mustShutDown(false), _task(nullptr), _dependencyDomain(),
	_hardwareCounters(), _instrumentationData()
{
	WorkerThreadBase::start();
}


inline WorkerThread::~WorkerThread()
{
}


inline Task *WorkerThread::getTask()
{
	return _task;
}


inline void WorkerThread::setTask(Task *task)
{
	assert(_task == nullptr);
	_task = task;
}


inline DependencyDomain const *WorkerThread::getDependencyDomain() const
{
	return &_dependencyDomain;
}

inline DependencyDomain *WorkerThread::getDependencyDomain()
{
	return &_dependencyDomain;
}


inline HardwareCountersThreadLocalData &WorkerThread::getHardwareCounters()
{
	return _hardwareCounters;
}

inline Instrument::ThreadLocalData &WorkerThread::getInstrumentationData()
{
	return _instrumentationData;
}


inline void WorkerThread::handleTask(CPU *cpu, Task *task)
{
	assert(task != nullptr);
	
	// Save current task
	Task *oldTask = _task;
	assert(task != oldTask);
	
	// Run the task
	_task = task;
	handleTask(cpu);
	
	// Restore the initial task
	_task = oldTask;
}


inline void WorkerThread::signalShutdown()
{
	_mustShutDown = true;
}

inline bool WorkerThread::hasPendingShutdown()
{
	return _mustShutDown;
}

inline WorkerThread *WorkerThread::getCurrentWorkerThread()
{
	WorkerThreadBase *thread = WorkerThreadBase::getCurrentWorkerThread();
	
	if (thread == nullptr) {
		return nullptr;
	} else if (typeid(*thread) == typeid(WorkerThread)) {
		return static_cast<WorkerThread *> (thread);
	} else {
		return nullptr;
	}
}



#ifndef NDEBUG
namespace ompss_debug {
	__attribute__((weak)) WorkerThread *getCurrentWorkerThread()
	{
		WorkerThread *current = WorkerThread::getCurrentWorkerThread();
		
		if (current == nullptr) {
			return (WorkerThread *) ~0UL;
		} else {
			return current;
		}
	}
}
#endif


#include "performance/HardwareCountersThreadLocalDataImplementation.hpp"


#endif // WORKER_THREAD_IMPLEMENTATION_HPP
