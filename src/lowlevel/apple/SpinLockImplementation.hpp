#ifndef APPLE_SPIN_LOCK_IMPLEMENTATION_HPP
#define APPLE_SPIN_LOCK_IMPLEMENTATION_HPP


#ifndef SPIN_LOCK_HPP
	#error Include SpinLock.h instead
#endif


inline SpinLock::SpinLock()
	: _lock(0)
{
}

inline SpinLock::~SpinLock()
{
}

inline void SpinLock::lock()
{
	OSSpinLockLock(&_lock);
}

inline bool SpinLock::tryLock()
{
	return OSSpinLockTry(&_lock);
}

inline void SpinLock::unlock()
{
	OSSpinLockUnlock(&_lock);
}


#endif // APPLE_SPIN_LOCK_IMPLEMENTATION_HPP
