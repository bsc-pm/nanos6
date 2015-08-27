#ifndef WORKER_THREAD_HPP
#define WORKER_THREAD_HPP


#include "lowlevel/ConditionVariable.hpp"

#include "CPU.hpp"

#include <pthread.h>


class Task;
class ThreadManager;
class SchedulingDecisionPlaceholder;


class WorkerThread {
	//! This condition variable is used for suspending and resuming the thread
	ConditionVariable _suspensionConditionVariable;
	
	//! The CPU assigned to this thread. Volatile, since the thread could migrate while blocked.
	CPU * volatile _cpu;
	
	//! The underlying pthread
	pthread_t _pthread;
	
	//! The Task currently assigned to this thread
	Task *_task;
	
	//! Thread Local Storage variable to point back to the WorkerThread that is running the code
	static __thread WorkerThread *_currentWorkerThread;
	
	//! \brief Suspend the thread
	//!
	//! \returns true if it was pre-resumed
	inline bool suspend()
	{
		return _suspensionConditionVariable.wait();
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
	
	//! \brief get the hardware place currently assigned
	inline CPU *getHardwarePlace()
	{
		return _cpu;
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
