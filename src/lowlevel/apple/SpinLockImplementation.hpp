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
	OSSpinLockLock(&_lock);
	assertUnowned();
	setOwner();
}

inline bool SpinLock::tryLock()
{
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

#ifndef NDEBUG
inline bool SpinLock::isLockedByThisThread()
{
	return (_owner == ompss_debug::getCurrentWorkerThread());
}
#endif

#endif // APPLE_SPIN_LOCK_IMPLEMENTATION_HPP
