/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef WORKER_THREAD_IMPLEMENTATION_HPP
#define WORKER_THREAD_IMPLEMENTATION_HPP

#include <cassert>
#include <typeinfo>

#include "CPU.hpp"
#include "DependencyDomain.hpp"
#include "WorkerThread.hpp"
#include "WorkerThreadBase.hpp"

#include <InstrumentThreadManagement.hpp>


inline WorkerThread::WorkerThread(CPU *cpu)
	: WorkerThreadBase(cpu), _task(nullptr), _dependencyDomain(),
	_instrumentationData(), _hwCounters(), _replacementCount(0)
{
	_originalNumaNode = cpu->getNumaNodeId();
	Instrument::enterThreadCreation(/* OUT */ _instrumentationId, cpu->getInstrumentationId());
	WorkerThreadBase::start();
	Instrument::exitThreadCreation(_instrumentationId);
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

inline Task *WorkerThread::unassignTask()
{
	Task *currentTask = _task;
	_task = nullptr;

	return currentTask;
}

inline DependencyDomain const *WorkerThread::getDependencyDomain() const
{
	return &_dependencyDomain;
}

inline DependencyDomain *WorkerThread::getDependencyDomain()
{
	return &_dependencyDomain;
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

inline ThreadHardwareCounters &WorkerThread::getHardwareCounters()
{
	return _hwCounters;
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


inline bool WorkerThread::isTaskReplaceable() const
{
	return (_replacementCount < _maxReplaceCount);
}

inline void WorkerThread::replaceTask(Task *task)
{
	++_replacementCount;
	assert(_replacementCount <= _maxReplaceCount);

	_task = task;
}

inline void WorkerThread::restoreTask(Task *task)
{
	--_replacementCount;

	_task = task;
}


#endif // WORKER_THREAD_IMPLEMENTATION_HPP
