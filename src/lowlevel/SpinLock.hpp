#ifndef SPIN_LOCK_HPP
#define SPIN_LOCK_HPP

#ifdef __APPLE__
	#include <libkern/OSAtomic.h>
	#define SPINLOCK_INTERNAL_TYPE OSSpinLock
#else
	#include <pthread.h>
	#define SPINLOCK_INTERNAL_TYPE pthread_spinlock_t
#endif


class SpinLock {
private:
	SpinLock operator=(const SpinLock &) = delete;
	SpinLock(const SpinLock & ) = delete;
	
	SPINLOCK_INTERNAL_TYPE _lock;
	
public:
	inline SpinLock();
	inline ~SpinLock();
	inline void lock();
	inline bool tryLock();
	inline void unlock();
};


#ifdef __APPLE__
	#include "apple/SpinLockImplementation.hpp"
#else
	#include "posix/SpinLockImplementation.hpp"
#endif

#undef SPINLOCK_INTERNAL_TYPE


#endif // SPIN_LOCK_HPP
