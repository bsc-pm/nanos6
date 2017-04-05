#ifndef WORKER_THREAD_HPP
#define WORKER_THREAD_HPP


#include "CPU.hpp"
#include "EssentialThreadEnvironment.hpp"
#include "Thread.hpp"

#include <atomic>
#include <cassert>


class Task;
class ThreadManager;
class SchedulingDecisionPlaceholder;


class WorkerThread : public Thread, public EssentialThreadEnvironment {
private:
	//! The CPU on which this thread is running.
	CPU *_cpu;
	
	//! The CPU to which the thread transitions the next time it resumes. Atomic since this is changed by other threads.
	std::atomic<CPU *> _cpuToBeResumedOn;
	
	//! Indicates that it is time for this thread to participate in the shutdown process
	std::atomic<bool> _mustShutDown;
	
	//! Thread Local Storage variable to point back to the WorkerThread that is running the code
	static __thread WorkerThread *_currentWorkerThread;
	
	void handleTask();
	
	//! Only the thread manager is suposed to call the suspend and resume methods. Any other use must go through it.
	friend class ThreadManager;
	
public:
	WorkerThread(CPU * cpu);
	virtual ~WorkerThread()
	{
	}
	 
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
	
	//! \brief get the hardware place currently assigned
	inline CPU *getComputePlace()
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
