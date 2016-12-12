#include <cassert>

#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>

#include "Thread.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


static void *thread_body_wrapper(void *parameter)
{
	Thread *thread = (Thread *) parameter;
	assert(thread != nullptr);
	thread->setTid(syscall(SYS_gettid));
	
	return thread->body();
}


void Thread::start(pthread_attr_t const *pthreadAttr)
{
	int rc = pthread_create(&_pthread, pthreadAttr, &thread_body_wrapper, this);
	FatalErrorHandler::handle(rc, " when creating a pthread");
}
