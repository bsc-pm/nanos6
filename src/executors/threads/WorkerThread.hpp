#ifndef WORKER_THREAD_HPP
#define WORKER_THREAD_HPP


#include "DependencyDomain.hpp"
#include "lowlevel/ConditionVariable.hpp"

#include "CPU.hpp"

#include <atomic>
#include <cassert>
#include <deque>

#include <pthread.h>


class Task;
class ThreadManager;
class SchedulingDecisionPlaceholder;


class WorkerThread {
public:
	typedef std::deque<Task *> satisfied_originator_list_t;
	
private:
	//! This condition variable is used for suspending and resuming the thread
	ConditionVariable _suspensionConditionVariable;
	
	//! The CPU on which this thread is running.
	CPU *_cpu;
	
	//! The CPU to which the thread transitions the next time it resumes. Atomic since this is changed by other threads.
	std::atomic<CPU *> _cpuToBeResumedOn;
	
	//! Indicates that it is time for this thread to participate in the shutdown process
	std::atomic<bool> _mustShutDown;
	
	//! The underlying pthread
	pthread_t _pthread;
	
	//! The Task currently assigned to this thread
	Task *_task;
	
	//! Dependency domain of the tasks instantiated by this thread
	DependencyDomain _dependencyDomain;
	
	//! Tasks whose accesses have been satified after ending a task
	satisfied_originator_list_t _satisfiedAccessOriginators;
	
	//! Thread Local Storage variable to point back to the WorkerThread that is running the code
	static __thread WorkerThread *_currentWorkerThread;
	
	//! \brief Suspend the thread
	inline void suspend()
	{
		assert(this == _currentWorkerThread);
		_suspensionConditionVariable.wait();
	}
	
	//! \brief Resume the thread
	inline void resume()
	{
		_suspensionConditionVariable.signal();
	}
	
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
	
	inline void exit()
	{
		pthread_exit(nullptr);
	}
	
	void handleTask();
	
	//! Only the thread manager is suposed to call the suspend and resume methods. Any other use must go through it.
	friend class ThreadManager;
	
public:
	WorkerThread(CPU * cpu);
	
	//! \brief handle a task
	//! This method is here to cover the case in which a task is run within the execution of another in the same thread
	inline void handleTask(Task *task)
	{
		assert(task != nullptr);
		
		// Save current task
		Task *oldTask = _task;
		assert(task != oldTask);
		
		// Run the task
		_task = task;
		handleTask();
		
		// Restore the initial task
		_task = oldTask;
	}
	
	//! \brief code that the thread executes
	void *body();
	
	inline int getCpuId()
	{
		return _cpu->_systemCPUId;
	}
	
	//! \brief get the currently assigned task to this thread
	inline Task *getTask()
	{
		return _task;
	}
	
	//! \brief set the task that this thread must run when it is resumed
	//!
	//! \param[in] task the task that the thread will run when it is resumed
	inline void setTask(Task *task)
	{
		assert(_task == nullptr);
		_task = task;
	}
	
	//! \brief get the hardware place currently assigned
	inline CPU *getHardwarePlace()
	{
		return _cpu;
	}
	
	
	//! \brief turn on the flag to start the shutdown process
	inline void signalShutdown()
	{
		_mustShutDown = true;
	}
	
	//! \brief get the thread shutdown flag
	inline bool hasPendingShutdown()
	{
		return _mustShutDown;
	}
	
	//! \brief Retrieves the dependency domain used to calculate the dependencies of the tasks instantiated by this thread
	DependencyDomain const *getDependencyDomain() const
	{
		return &_dependencyDomain;
	}
	
	//! \brief Retrieves the dependency domain used to calculate the dependencies of the tasks instantiated by this thread
	DependencyDomain *getDependencyDomain()
	{
		return &_dependencyDomain;
	}
	
	//! \brief Retrieves a reference to the thread-local list of tasks that have had one of their accesses satisfied when removing a task
	satisfied_originator_list_t &getSatisfiedOriginatorsReference()
	{
		return _satisfiedAccessOriginators;
	}
	
	//! \brief returns the WorkerThread that runs the call
	static inline WorkerThread *getCurrentWorkerThread()
	{
		return _currentWorkerThread;
	}
	
};



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


#endif // WORKER_THREAD_HPP
