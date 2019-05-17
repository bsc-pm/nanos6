/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef WORKER_THREAD_HPP
#define WORKER_THREAD_HPP


#include <atomic>

#include "DependencyDomain.hpp"
#include "WorkerThreadBase.hpp"
#include "instrument/stats/InstrumentHardwareCountersThreadLocalData.hpp"

#include <InstrumentThreadLocalData.hpp>
#include <ThreadHardwareCounters.hpp>


struct CPU;
class Task;
class ThreadManager;
class WorkerThreadRunner;


class WorkerThread : public WorkerThreadBase {
private:
	//! Indicates that it is time for this thread to participate in the shutdown process
	std::atomic<bool> _mustShutDown;
	
	//! The Task currently assigned to this thread
	Task *_task;
	
	//! The NUMA node to which this thread should have affinity
	size_t _originalNumaNode;
	
	//! Dependency domain of the tasks instantiated by this thread
	DependencyDomain _dependencyDomain;
	
	HardwareCountersThreadLocalData _hardwareCounters;
	
	Instrument::ThreadLocalData _instrumentationData;
	
	ThreadHardwareCounters _threadCounters;
	
	void initialize();
	void handleTask(CPU *cpu);
	
	friend class ThreadManager;
	friend class WorkerThreadRunner;
	
	
public:
	WorkerThread() = delete;
	
	inline WorkerThread(CPU * cpu);
	
	inline virtual ~WorkerThread();
	
	//! \brief get the currently assigned task to this thread
	inline Task *getTask();
	
	//! \brief set the task that this thread must run when it is resumed
	//!
	//! \param[in] task the task that the thread will run when it is resumed
	inline void setTask(Task *task);
	
	inline size_t getOriginalNumaNode() const
	{
		return _originalNumaNode;
	}
	
	//! \brief Retrieves the dependency domain used to calculate the dependencies of the tasks instantiated by this thread
	inline DependencyDomain const *getDependencyDomain() const;
	
	//! \brief Retrieves the dependency domain used to calculate the dependencies of the tasks instantiated by this thread
	inline DependencyDomain *getDependencyDomain();
	
	inline HardwareCountersThreadLocalData &getHardwareCounters();
	
	inline Instrument::ThreadLocalData &getInstrumentationData();
	
	//! \brief handle a task
	//! This method is here to cover the case in which a task is run within the execution of another in the same thread
	inline void handleTask(CPU *cpu, Task *task);
	
	//! \brief code that the thread executes
	virtual void body();
	
	
	//! \brief turn on the flag to start the shutdown process
	inline void signalShutdown();
	
	//! \brief get the thread shutdown flag
	inline bool hasPendingShutdown();
	
	//! \brief returns the WorkerThread that runs the call
	static inline WorkerThread *getCurrentWorkerThread();
	
	//! \brief Returns the thread's hardware counter structures
	inline ThreadHardwareCounters *getThreadHardwareCounters();
	
};



#ifndef NDEBUG
namespace ompss_debug {
	__attribute__((weak)) WorkerThread *getCurrentWorkerThread();
}
#endif


#include "WorkerThreadImplementation.hpp"


#endif // WORKER_THREAD_HPP
