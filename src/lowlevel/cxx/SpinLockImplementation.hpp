/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef CXX_SPIN_LOCK_IMPLEMENTATION_HPP
#define CXX_SPIN_LOCK_IMPLEMENTATION_HPP


#ifndef SPIN_LOCK_READS_BETWEEN_CMPXCHG
#define SPIN_LOCK_READS_BETWEEN_CMPXCHG 1000
#endif


#ifndef SPIN_LOCK_HPP
	#error Include SpinLock.hpp instead
#include "../SpinLock.hpp"
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
	
	bool expected = false;
	while (!_lock.compare_exchange_weak(expected, true, std::memory_order_acquire)) {
		int spinsLeft = SPIN_LOCK_READS_BETWEEN_CMPXCHG;
		do {
			expected = _lock.load(std::memory_order_relaxed);
			spinsLeft--;
		} while (expected && (spinsLeft > 0));

		expected = false;
	}
	
	DEBUG_KIND::assertUnowned();
	DEBUG_KIND::setOwner();
}

template <class DEBUG_KIND>
inline bool CustomizableSpinLock<DEBUG_KIND>::tryLock()
{
	DEBUG_KIND::assertNotCurrentOwner();
	
	bool expected = false;
	bool success = _lock.compare_exchange_strong(expected, true, std::memory_order_acquire);
	
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
	_lock.store(false, std::memory_order_release);
}


#endif // CXX_SPIN_LOCK_IMPLEMENTATION_HPP
