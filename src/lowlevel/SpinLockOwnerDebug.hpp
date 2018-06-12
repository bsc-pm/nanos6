/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef SPIN_LOCK_OWNER_DEBUG_HPP
#define SPIN_LOCK_OWNER_DEBUG_HPP


#include <cassert>


class WorkerThread;
namespace ompss_debug {
	__attribute__((weak)) WorkerThread *getCurrentWorkerThread();
}


class SpinLockOwnerDebug {
private:
	WorkerThread *_owner;
	
public:
	inline SpinLockOwnerDebug()
		: _owner(nullptr)
	{
	}
	
	inline void assertCurrentOwner(__attribute__((unused)) bool ignoreOwner)
	{
		assert(ignoreOwner || (_owner == ompss_debug::getCurrentWorkerThread()));
	}
	
	inline void assertUnowned()
	{
		assert(_owner == nullptr);
	}
	
	inline void assertUnownedOrCurrentOwner(__attribute__((unused)) bool ignoreOwner)
	{
		assert( ignoreOwner || (_owner == nullptr) || (_owner == ompss_debug::getCurrentWorkerThread()) ) ;
	}
	
	inline void assertNotCurrentOwner()
	{
		assert(_owner != ompss_debug::getCurrentWorkerThread());
	}
	
	inline void setOwner()
	{
		_owner = ompss_debug::getCurrentWorkerThread();
	}
	
	inline void unsetOwner()
	{
		_owner = nullptr;
	}
	
	inline bool isLockedByThisThread()
	{
		return (_owner != nullptr);
	}
	
};


#endif // SPIN_LOCK_OWNER_DEBUG_HPP
