/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef WORKER_THREAD_HPP
#define WORKER_THREAD_HPP

#include <random>

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

	//! Count for the number of tasks replaced in this thread
	size_t _replacementCount;
	static constexpr size_t _maxReplaceCount = 16;

	void initialize();
	void handleTask(CPU *cpu, bool);
	void executeTask(CPU *cpu);

	friend class ThreadManager;
	friend class WorkerThreadRunner;
	friend class Throttle;

	//! Immediate successor probability generator
	//! NOTE: Need one per WorkerThread as we want the distribution to act
	//! in chain within each thread
	std::default_random_engine _ISGenerator;
	std::uniform_real_distribution<float> _ISDistribution;

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

	//! \brief Returns if the task on the thread can currently be replaced
	inline bool isTaskReplaceable() const;

	//! \brief Replaces the current task inside the thread
	inline void replaceTask(Task *task);

	//! \brief Restores a task that was previously assigned to this thread
	inline void restoreTask(Task *task);
};


#ifndef NDEBUG
namespace ompss_debug {
	__attribute__((weak)) WorkerThread *getCurrentWorkerThread();
}
#endif


#include "WorkerThreadImplementation.hpp"


#endif // WORKER_THREAD_HPP
