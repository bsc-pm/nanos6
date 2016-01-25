#ifndef CXX_SPIN_LOCK_IMPLEMENTATION_HPP
#define CXX_SPIN_LOCK_IMPLEMENTATION_HPP


#ifndef SPIN_LOCK_HPP
	#error Include SpinLock.hpp instead
#include "../SpinLock.hpp"
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
	bool expected;
	do {
		expected = false;
	} while (!_lock.compare_exchange_weak(expected, true, std::memory_order_acquire));
	
	assertUnowned();
	setOwner();
}

inline bool SpinLock::tryLock()
{
	bool expected = false;
	bool success = _lock.compare_exchange_strong(expected, true, std::memory_order_acquire);
	
	if (success) {
		assertUnowned();
		setOwner();
	}
	
	return success;
}

inline void SpinLock::unlock()
{
	assertCurrentOwner();
	unsetOwner();
	_lock.store(false, std::memory_order_release);
}

#ifndef NDEBUG
inline bool SpinLock::isLockedByThisThread()
{
	return (_owner == ompss_debug::getCurrentWorkerThread());
}
#endif


#endif // CXX_SPIN_LOCK_IMPLEMENTATION_HPP
