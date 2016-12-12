#ifndef THREAD_HPP
#define THREAD_HPP


#include <pthread.h>
#include <sys/types.h>


class Thread {
protected:
	//! The underlying pthread
	pthread_t _pthread;
	pid_t _tid;
	
	
	inline void exit()
	{
		pthread_exit(nullptr);
	}
	
public:
	Thread()
	{
	}
	
	virtual ~Thread()
	{
	}
	
	// WARNING: This should be only called by the thread initialization code
	inline void setTid(pid_t tid)
	{
		_tid = tid;
	}
	
	void start(pthread_attr_t const *pthreadAttr);
	
	//! \brief code that the thread executes
	virtual void *body() = 0;
	
};


#endif // THREAD_HPP
