/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef SPIN_LOCK_NO_DEBUG_HPP
#define SPIN_LOCK_NO_DEBUG_HPP


class SpinLockNoDebug {
public:
	inline SpinLockNoDebug()
	{
	}
	
	inline void willLock()
	{
	}
	
	inline void assertCurrentOwner(__attribute__((unused)) bool ignoreOwner)
	{
	}
	
	inline void assertUnowned()
	{
	}
	
	inline void assertUnownedOrCurrentOwner(__attribute__((unused)) bool ignoreOwner)
	{
	}
	
	inline void assertNotCurrentOwner()
	{
	}
	
	inline void setOwner()
	{
	}
	
	inline void unsetOwner()
	{
	}
	
};


#endif // SPIN_LOCK_NO_DEBUG_HPP
