#ifndef THREAD_HPP
#define THREAD_HPP


#include <pthread.h>


class Thread {
protected:
	//! The underlying pthread
	pthread_t _pthread;
	
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
	
	void start(pthread_attr_t const *pthreadAttr);
	
	//! \brief code that the thread executes
	virtual void *body() = 0;
	
};


#endif // THREAD_HPP
