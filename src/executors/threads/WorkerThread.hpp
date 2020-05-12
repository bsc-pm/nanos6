/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef WORKER_THREAD_HPP
#define WORKER_THREAD_HPP

#include "DependencyDomain.hpp"
#include "WorkerThreadBase.hpp"
#include "hardware-counters/ThreadHardwareCounters.hpp"

#include <InstrumentThreadLocalData.hpp>


class CPU;
class Task;
class ThreadManager;
class WorkerThreadRunner;


class WorkerThread : public WorkerThreadBase {
private:
	//! The Task currently assigned to this thread
	Task *_task;

	//! The NUMA node to which this thread should have affinity
	size_t _originalNumaNode;

	//! Dependency domain of the tasks instantiated by this thread
	DependencyDomain _dependencyDomain;

	Instrument::ThreadLocalData _instrumentationData;

	ThreadHardwareCounters _hwCounters;

	void initialize();
	void handleTask(CPU *cpu);

	friend class ThreadManager;
	friend class WorkerThreadRunner;
	friend class Throttle;

public:
	WorkerThread() = delete;

	inline WorkerThread(CPU *cpu);

	inline virtual ~WorkerThread();

	//! \brief Get the currently assigned task to this thread
	inline Task *getTask();

	//! \brief Set the task that this thread must run when it is resumed
	//!
	//! \param[in] task the task that the thread will run when it is resumed
	inline void setTask(Task *task);

	//! \brief Unattach the current task assigned to the thread
	//!
	//! \return A valid task or nullptr
	inline Task *unassignTask();

	inline size_t getOriginalNumaNode() const
	{
		return _originalNumaNode;
	}

	//! \brief Retrieves the dependency domain used to calculate the dependencies of the tasks instantiated by this thread
	inline DependencyDomain const *getDependencyDomain() const;

	//! \brief Retrieves the dependency domain used to calculate the dependencies of the tasks instantiated by this thread
	inline DependencyDomain *getDependencyDomain();

	inline Instrument::ThreadLocalData &getInstrumentationData();

	//! \brief handle a task
	//! This method is here to cover the case in which a task is run within the execution of another in the same thread
	inline void handleTask(CPU *cpu, Task *task);

	//! \brief code that the thread executes
	virtual void body();

	//! \brief returns the WorkerThread that runs the call
	static inline WorkerThread *getCurrentWorkerThread();

	//! \brief Returns the thread's hardware counter structures
	inline ThreadHardwareCounters &getHardwareCounters();

};



#ifndef NDEBUG
namespace ompss_debug {
	__attribute__((weak)) WorkerThread *getCurrentWorkerThread();
}
#endif


#include "WorkerThreadImplementation.hpp"


#endif // WORKER_THREAD_HPP
