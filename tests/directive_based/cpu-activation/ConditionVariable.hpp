/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef CONDITION_VARIABLE_HPP
#define CONDITION_VARIABLE_HPP


#ifndef NDEBUG
#include <iostream>
#endif

#ifndef NDEBUG
#include <pthread.h>
#endif

#include <cassert>


#if __cplusplus >= 201103L

#include <atomic>
#include <condition_variable>
#include <mutex>

class ConditionVariable {
	bool _signaled;
	
	std::mutex _mutex;
	std::condition_variable _condVar;
	
	#ifndef NDEBUG
		std::atomic<long> _owner;
	#endif
	
public:
	ConditionVariable(const ConditionVariable &) = delete;
	ConditionVariable operator=(const ConditionVariable &) = delete;
	
	ConditionVariable()
		: _signaled(false)
		#ifndef NDEBUG
			, _owner(0)
		#endif
	{
	}
	
	
	//! \brief Wait on the contition variable until signaled
	void wait()
	{
		#ifndef NDEBUG
			{
				if (_owner == 0) {
					long expected = 0;
					long want = (long) pthread_self();
					assert(_owner.compare_exchange_strong(expected, want));
				} else {
					long currentThread = (long) pthread_self();
					assert(_owner == currentThread);
				}
			}
		#endif
		
		std::unique_lock<std::mutex> lock(_mutex);
		while (!_signaled) {
			_condVar.wait(lock);
		}
		
		// Initialize for next time
		_signaled = false;
	}
	
	//! \brief Signal the condition variable to wake up a thread that is waiting or will wait on it
	void signal()
	{
		#ifndef NDEBUG
			{
				long currentThread = (long) pthread_self();
				assert(_owner != currentThread);
			}
		#endif
		
		{
			std::unique_lock<std::mutex> lock(_mutex);
			assert(_signaled == false);
			_signaled = true;
		}
		
		_condVar.notify_one();
	}
	
	bool isPresignaled()
	{
		return _signaled;
	}
	
	void clearPresignal()
	{
		assert(_signaled);
		_signaled = false;
	}
	
};

#elif USE_BLOCKING_API

// Unfortunately this changes the expected behaviour of the test

// C++ 03

#include <nanos6/blocking.h>

#include <Atomic.hpp>

class ConditionVariable {
	Atomic<bool> _signaled;
	Atomic<void *> _blockingContext;
	
	#ifndef NDEBUG
		Atomic<long> _owner;
	#endif
	
private:
	ConditionVariable(const ConditionVariable &);
	ConditionVariable operator=(const ConditionVariable &);
	
public:
	ConditionVariable()
		: _signaled(false), _blockingContext(0)
		#ifndef NDEBUG
			, _owner(0)
		#endif
	{
	}
	
	
	//! \brief Wait on the contition variable until signaled
	void wait()
	{
		#ifndef NDEBUG
			{
				if (_owner == 0) {
					long expected = 0;
					long want = (long) pthread_self();
				} else {
					long currentThread = (long) pthread_self();
					assert(_owner == currentThread);
				}
			}
		#endif
		
		// Initialize for next time
		_signaled.store(false);
		
		_blockingContext.store(nanos6_get_current_blocking_context());
		nanos6_block_current_task(_blockingContext);
	}
	
	
	//! \brief Signal the condition variable to wake up a thread that is waiting or will wait on it
	void signal()
	{
		#ifndef NDEBUG
			{
				long currentThread = (long) pthread_self();
				assert(_owner != currentThread);
			}
		#endif
		
		assert(_signaled.load() == false);
		_signaled.store(true);
		
		assert(_blockingContext != 0);
		void *blockingContext = _blockingContext;
		_blockingContext.store(0);
		
		nanos6_unblock_task(blockingContext);
	}
	
	bool isPresignaled()
	{
		return _signaled;
	}
	
	void clearPresignal()
	{
		assert(_signaled);
		_signaled.store(false);
	}
	
};


#else

// C++03

#include <pthread.h>


class ConditionVariable {
	bool _signaled;
	
	pthread_mutex_t _mutex;
	pthread_cond_t _condVar;
	
	ConditionVariable(const ConditionVariable &);
	ConditionVariable operator=(const ConditionVariable &);
	
public:
	ConditionVariable()
		: _signaled(false)
	{
		pthread_mutex_init(&_mutex, 0);
		pthread_cond_init(&_condVar, 0);
	}
	
	
	//! \brief Wait on the contition variable until signaled
	void wait()
	{
		pthread_mutex_lock(&_mutex);
		while (!_signaled) {
			pthread_cond_wait(&_condVar, &_mutex);
		}
		
		// Initialize for next time
		_signaled = false;
		pthread_mutex_unlock(&_mutex);
	}
	
	//! \brief Signal the condition variable to wake up a thread that is waiting or will wait on it
	void signal()
	{
		pthread_mutex_lock(&_mutex);
		assert(_signaled == false);
		_signaled = true;
		
		pthread_cond_signal(&_condVar);
		pthread_mutex_unlock(&_mutex);
	}
	
	bool isPresignaled()
	{
		return _signaled;
	}
	
	void clearPresignal()
	{
		assert(_signaled);
		_signaled = false;
	}
	
};


#endif

#endif // CONDITION_VARIABLE_HPP
