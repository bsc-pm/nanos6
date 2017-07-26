/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef GLOBAL_LOCK_HPP
#define GLOBAL_LOCK_HPP


#include "lowlevel/SpinLock.hpp"


class GlobalLock {
private:
	static SpinLock _lock;
	
public:
	static void lock()
	{
		_lock.lock();
	}
	
	static void unlock()
	{
		_lock.unlock();
	}
};


#endif // GLOBAL_LOCK_HPP
