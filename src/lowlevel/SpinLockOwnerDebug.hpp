/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef SPIN_LOCK_OWNER_DEBUG_HPP
#define SPIN_LOCK_OWNER_DEBUG_HPP


#include <cassert>


namespace ompss_debug {
	void *getCurrentThread();
}


class SpinLockOwnerDebug {
private:
	void *_owner;
	
public:
	inline SpinLockOwnerDebug()
		: _owner(nullptr)
	{
	}
	
	inline void willLock()
	{
	}
	
	inline void assertCurrentOwner(__attribute__((unused)) bool ignoreOwner)
	{
		assert(ignoreOwner || (_owner == ompss_debug::getCurrentThread()) || (ompss_debug::getCurrentThread() == nullptr));
	}
	
	inline void assertUnowned()
	{
		assert(_owner == nullptr);
	}
	
	inline void assertUnownedOrCurrentOwner(__attribute__((unused)) bool ignoreOwner)
	{
		assert( ignoreOwner || (_owner == nullptr) || (_owner == ompss_debug::getCurrentThread()) ) ;
	}
	
	inline void assertNotCurrentOwner()
	{
		assert((_owner != ompss_debug::getCurrentThread()) || (ompss_debug::getCurrentThread() == nullptr));
	}
	
	inline void setOwner()
	{
		_owner = ompss_debug::getCurrentThread();
	}
	
	inline void unsetOwner()
	{
		_owner = nullptr;
	}
	
	inline bool isLockedByThisThread()
	{
		return (_owner == ompss_debug::getCurrentThread());
	}
	
};


#endif // SPIN_LOCK_OWNER_DEBUG_HPP
