/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef USER_MUTEX_HPP
#define USER_MUTEX_HPP


#include "lowlevel/SpinLock.hpp"

#include <atomic>
#include <cassert>
#include <deque>
#include <mutex>


class Task;


class UserMutex {
	//! \brief The user mutex state
	std::atomic<bool> _userMutex;
	
	//! \brief The spin lock that protects the queue of tasks blocked on this user-side mutex
	SpinLock _blockedTasksLock;
	
	//! \brief The list of tasks blocked on this user-side mutex
	std::deque<Task *> _blockedTasks;
	
public:
	//! \brief Initialize the mutex
	//!
	//! \param[in] initialState true if the mutex must be initialized in the locked state
	inline UserMutex(bool initialState)
	: _userMutex(initialState), _blockedTasksLock(), _blockedTasks()
	{
	}
	
	//! \brief Try to lock
	//!
	//! \returns true if the user-lock has been locked successfully, false otherwise
	inline bool tryLock()
	{
		bool expected = false;
		bool successful = _userMutex.compare_exchange_strong(expected, true);
		assert(expected != successful);
		return successful;
	}
	
	//! \brief Try to lock of queue the task
	//!
	//! \param[in] task The task that will be queued if the lock cannot be acquired
	//!
	//! \returns true if the lock has been acquired, false if not and the task has been queued
	inline bool lockOrQueue(Task *task)
	{
		std::lock_guard<SpinLock> guard(_blockedTasksLock);
		if (tryLock()) {
			return true;
		} else {
			_blockedTasks.push_back(task);
			return false;
		}
	}
	
	inline Task *dequeueOrUnlock()
	{
		std::lock_guard<SpinLock> guard(_blockedTasksLock);
		
		if (_blockedTasks.empty()) {
			_userMutex = false;
			return nullptr;
		}
		
		Task *releasedTask = _blockedTasks.front();
		_blockedTasks.pop_front();
		assert(releasedTask != nullptr);
		
		return releasedTask;
	}
};


#endif // USER_MUTEX_HPP
