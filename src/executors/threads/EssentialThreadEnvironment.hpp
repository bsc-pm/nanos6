#ifndef ESSENTIAL_THREAD_ENVIRONMENT_HPP
#define ESSENTIAL_THREAD_ENVIRONMENT_HPP


#include "DependencyDomain.hpp"
#include "lowlevel/ConditionVariable.hpp"

#include <pthread.h>


class Task;


class EssentialThreadEnvironment {
protected:
	//! This condition variable is used for suspending and resuming the thread
	ConditionVariable _suspensionConditionVariable;
	
	//! The Task currently assigned to this thread
	Task *_task;
	
	//! Dependency domain of the tasks instantiated by this thread
	DependencyDomain _dependencyDomain;
	
	//! The underlying pthread
	pthread_t _pthread;
	
	
public:
	EssentialThreadEnvironment()
		: _suspensionConditionVariable(), _task(nullptr), _dependencyDomain()
	{
	}
	
	virtual ~EssentialThreadEnvironment()
	{
	}
	
	//! \brief Suspend the thread
	inline void suspend()
	{
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
	
};


#endif // ESSENTIAL_THREAD_ENVIRONMENT_HPP
