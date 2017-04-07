#ifndef WORKER_THREAD_HPP
#define WORKER_THREAD_HPP


#include "CPU.hpp"
#include "DependencyDomain.hpp"
#include "WorkerThreadBase.hpp"

#include <atomic>
#include <cassert>


class Task;
class ThreadManager;
class SchedulingDecisionPlaceholder;


class WorkerThread : public WorkerThreadBase {
private:
	//! Indicates that it is time for this thread to participate in the shutdown process
	std::atomic<bool> _mustShutDown;
	
	//! The Task currently assigned to this thread
	Task *_task;
	
	//! Dependency domain of the tasks instantiated by this thread
	DependencyDomain _dependencyDomain;
	
	void initialize();
	void handleTask();
	
	//! Only the thread manager is supposed to call the suspend and resume methods. Any other use must go through it.
	friend class ThreadManager;
	
	
public:
	WorkerThread() = delete;
	
	inline WorkerThread(CPU * cpu)
		: WorkerThreadBase(cpu), _mustShutDown(false), _task(nullptr), _dependencyDomain()
	{
		start();
	}
	
	virtual ~WorkerThread()
	{
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
	virtual void *body();
	
	
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
		return static_cast<WorkerThread *> (WorkerThreadBase::getCurrentWorkerThread());
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
