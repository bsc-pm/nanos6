/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef MULTI_CONDITION_VARIABLE_HPP
#define MULTI_CONDITION_VARIABLE_HPP


#include <atomic>
#include <cassert>
#include <condition_variable>
#include <mutex>


class MultiConditionVariable {
	const size_t _waiters;
	size_t _signaled;

	std::mutex _mutex;
	std::condition_variable _condVar;

public:
	MultiConditionVariable(const MultiConditionVariable &) = delete;
	MultiConditionVariable operator=(const MultiConditionVariable &) = delete;

	MultiConditionVariable(size_t waiters) :
		_waiters(waiters), _signaled(0)
	{
	}

	//! \brief Wait on the condition variable until signaled
	void wait()
	{
		std::unique_lock<std::mutex> lock(_mutex);
		while (!_signaled) {
			_condVar.wait(lock);
		}

		// Decrease the signal counter
		--_signaled;
	}

	//! \brief Signal the condition variable
	//!
	//! This function signals the condition variable to wake up all threads that
	//! are waiting or will wait on it
	void signalAll()
	{
		{
			std::unique_lock<std::mutex> lock(_mutex);
			assert(_signaled == 0);
			_signaled += _waiters;
		}

		_condVar.notify_all();
	}
};


#endif // MULTI_CONDITION_VARIABLE_HPP
