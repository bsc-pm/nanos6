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
		std::unique_lock<std::mutex> lock(_mutex);
		assert(_signaled == false);
		_signaled = true;
		
		// The lock needs to cover the notification for the following scenario:
		// 1. Signaler enters the mutual exclusion, sets up the flag and exits the mutual exclusion
		// 2. Waiter enters the mutual exclusion, checks the flag, resets it, exits the mutual exclusion, and returns without blocking
		// 3. Waiter calls again the wait method: enters the mutual exclusion, checks the flag, and blocks on the condition variable
		// 4. The signaler now calls the condition variable notification (that was supposed to be sent togheter with setting the flag)
		// 5. The waiter gets (wrongly) woken up
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
