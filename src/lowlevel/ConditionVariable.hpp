#ifndef CONDITION_VARIABLE_HPP
#define CONDITION_VARIABLE_HPP


#include <cassert>
#include <condition_variable>
#include <mutex>

#ifndef NDEBUG
#include <iostream>
#endif


class ConditionVariable {
	bool _signaled;
	
	std::mutex _mutex;
	std::condition_variable _condVar;
	
	
public:
	ConditionVariable(const ConditionVariable &) = delete;
	ConditionVariable operator=(const ConditionVariable &) = delete;
	
	ConditionVariable()
		: _signaled(false)
	{
	}
	
	
	//! \brief Wait on the contition variable until signaled
	//!
	//! \returns true if it had already been signaled
	bool wait()
	{
		std::unique_lock<std::mutex> lock(_mutex);
		if (_signaled) {
			_signaled = false;
			return true;
		}
		
		_condVar.wait(lock);
		assert(_signaled);
		
		// Initialize for next time
		_signaled = false;
		
		return false;
	}
	
	//! \brief Signal the condition variable to wake up a thread that is waiting or will wait on it
	void signal()
	{
		{
			std::unique_lock<std::mutex> lock(_mutex);
			assert(_signaled == false);
			_signaled = true;
		}
		
		_condVar.notify_one();
	}
	
};


#endif // CONDITION_VARIABLE_HPP
