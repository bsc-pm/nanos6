/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef CONDITION_VARIABLE_HPP
#define CONDITION_VARIABLE_HPP


#include <atomic>
#include <cassert>
#include <condition_variable>
#include <mutex>

#ifndef NDEBUG
#include <iostream>
#endif

#ifndef NDEBUG
#include <pthread.h>
#endif


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
	
	
	//! \brief Wait on the condition variable until signaled
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


#endif // CONDITION_VARIABLE_HPP
