#ifndef POSIX_KERNEL_LEVEL_THREAD_HPP
#define POSIX_KERNEL_LEVEL_THREAD_HPP

#include <cassert>

#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>

#include "lowlevel/ConditionVariable.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


static void *kernel_level_thread_body_wrapper(void *parameter);


class KernelLevelThread {
protected:
	//! The underlying pthread
	pthread_t _pthread;
	pid_t _tid;
	
	//! This condition variable is used for suspending and resuming the thread
	ConditionVariable _suspensionConditionVariable;
	
	//! Thread Local Storage variable to point back to the KernelLevelThread that is running the code
	static __thread KernelLevelThread *_currentKernelLevelThread;
	
	inline void exit()
	{
		pthread_exit(nullptr);
	}
	
	inline void setCurrentKernelLevelThread()
	{
		_currentKernelLevelThread = this;
	}
	
	friend void *kernel_level_thread_body_wrapper(void *parameter);
	
	
public:
	KernelLevelThread()
	{
	}
	
	virtual ~KernelLevelThread()
	{
	}
	
	// WARNING: This should be only called by the thread initialization code
	inline void setTid(pid_t tid)
	{
		_tid = tid;
	}
	
	inline void start(pthread_attr_t const *pthreadAttr);
	
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
	
	//! \brief Wait for the thread to finish and join it
	inline void join();
	
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
	
	//! \brief code that the thread executes
	virtual void body() = 0;
	
	static inline KernelLevelThread *getCurrentKernelLevelThread()
	{
		return _currentKernelLevelThread;
	}
	
};


void *kernel_level_thread_body_wrapper(void *parameter)
{
	KernelLevelThread *thread = static_cast<KernelLevelThread *> (parameter);
	
	assert(thread != nullptr);
	thread->setTid(syscall(SYS_gettid));
	
	KernelLevelThread::_currentKernelLevelThread = thread;
	
	thread->body();
	
	return nullptr;
}


void KernelLevelThread::start(pthread_attr_t const *pthreadAttr)
{
	int rc = pthread_create(&_pthread, pthreadAttr, &kernel_level_thread_body_wrapper, this);
	FatalErrorHandler::handle(rc, " when creating a pthread");
}


void KernelLevelThread::join()
{
	int rc = pthread_join(_pthread, nullptr);
	FatalErrorHandler::handle(rc, " during shutdown when joining pthread ", _pthread);
}



#endif // POSIX_KERNEL_LEVEL_THREAD_HPP
