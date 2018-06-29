/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef APPLE_SPIN_LOCK_IMPLEMENTATION_HPP
#define APPLE_SPIN_LOCK_IMPLEMENTATION_HPP


#ifndef SPIN_LOCK_HPP
	#error Include SpinLock.h instead
#endif


template <class DEBUG_KIND>
inline CustomizableSpinLock<DEBUG_KIND>::CustomizableSpinLock()
	: _lock(0)
{
}

template <class DEBUG_KIND>
inline CustomizableSpinLock<DEBUG_KIND>::~CustomizableSpinLock()
{
	DEBUG_KIND::assertUnowned();
}

template <class DEBUG_KIND>
inline void CustomizableSpinLock<DEBUG_KIND>::lock()
{
	DEBUG_KIND::assertNotCurrentOwner();
	DEBUG_KIND::willLock();
	OSSpinLockLock(&_lock);
	DEBUG_KIND::assertUnowned();
	DEBUG_KIND::setOwner();
}

template <class DEBUG_KIND>
inline bool CustomizableSpinLock<DEBUG_KIND>::tryLock()
{
	DEBUG_KIND::assertNotCurrentOwner();
	bool success = OSSpinLockTry(&_lock);
	
	if (success) {
		DEBUG_KIND::assertUnowned();
		DEBUG_KIND::setOwner();
	}
	
	return success;
}

template <class DEBUG_KIND>
inline void CustomizableSpinLock<DEBUG_KIND>::unlock(bool ignoreOwner)
{
	DEBUG_KIND::assertCurrentOwner(ignoreOwner);
	DEBUG_KIND::unsetOwner();
	OSSpinLockUnlock(&_lock);
}


#endif // APPLE_SPIN_LOCK_IMPLEMENTATION_HPP
