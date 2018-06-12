/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef APPLE_SPIN_LOCK_IMPLEMENTATION_HPP
#define APPLE_SPIN_LOCK_IMPLEMENTATION_HPP


#ifndef SPIN_LOCK_HPP
	#error Include SpinLock.h instead
#endif


inline SpinLock::SpinLock()
	: _lock(0)
{
#ifndef NDEBUG
	_owner = nullptr;
#endif
}

inline SpinLock::~SpinLock()
{
	assertUnowned();
}

inline void SpinLock::lock()
{
	assertNotCurrentOwner();
	OSSpinLockLock(&_lock);
	assertUnowned();
	setOwner();
}

inline bool SpinLock::tryLock()
{
	assertNotCurrentOwner();
	bool success = OSSpinLockTry(&_lock);
	
	if (success) {
		assertUnowned();
		setOwner();
	}
	
	return success;
}

inline void SpinLock::unlock(bool ignoreOwner)
{
	assertCurrentOwner(ignoreOwner);
	unsetOwner();
	OSSpinLockUnlock(&_lock);
}


#endif // APPLE_SPIN_LOCK_IMPLEMENTATION_HPP
