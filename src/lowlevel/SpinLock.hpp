/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef SPIN_LOCK_HPP
#define SPIN_LOCK_HPP

#if USE_OLD_SPINLOCK_IMPLEMENTATION
	#ifdef __APPLE__
		#include <libkern/OSAtomic.h>
		#define SPINLOCK_INTERNAL_TYPE OSSpinLock
	#else
		#include <pthread.h>
		#define SPINLOCK_INTERNAL_TYPE pthread_spinlock_t
	#endif
#else
	#include <atomic>
	#define SPINLOCK_INTERNAL_TYPE std::atomic<bool>
#endif

#include <cassert>


#ifndef NDEBUG
class WorkerThread;
namespace ompss_debug {
	__attribute__((weak)) WorkerThread *getCurrentWorkerThread();
}
#endif


class SpinLock {
private:
	SpinLock operator=(const SpinLock &) = delete;
	SpinLock(const SpinLock & ) = delete;
	
	SPINLOCK_INTERNAL_TYPE _lock;
	
#ifndef NDEBUG
	WorkerThread *_owner;
#endif
	
	inline void assertCurrentOwner(__attribute__((unused)) bool ignoreOwner);
	inline void assertUnowned();
	inline void assertUnownedOrCurrentOwner(__attribute__((unused)) bool ignoreOwner);
	inline void assertNotCurrentOwner();
	inline void setOwner();
	inline void unsetOwner();
	
public:
	inline SpinLock();
	inline ~SpinLock();
	inline void lock();
	inline bool tryLock();
	inline void unlock(bool ignoreOwner = false);
#ifndef NDEBUG
	inline bool isLockedByThisThread();
#endif
};


#ifndef NDEBUG
inline void SpinLock::assertCurrentOwner(__attribute__((unused)) bool ignoreOwner)
{
	assert(ignoreOwner || (_owner == ompss_debug::getCurrentWorkerThread()));
}

inline void SpinLock::assertUnowned()
{
	assert(_owner == nullptr);
}

inline void SpinLock::assertUnownedOrCurrentOwner(__attribute__((unused)) bool ignoreOwner)
{
	assert( ignoreOwner || (_owner == nullptr) || (_owner == ompss_debug::getCurrentWorkerThread()) ) ;
}

inline void SpinLock::assertNotCurrentOwner()
{
	assert(_owner != ompss_debug::getCurrentWorkerThread());
}

inline void SpinLock::setOwner()
{
	_owner = ompss_debug::getCurrentWorkerThread();
}

inline void SpinLock::unsetOwner()
{
	_owner = nullptr;
}

inline bool SpinLock::isLockedByThisThread()
{
	return (_owner != nullptr);
}
#else
inline void SpinLock::assertCurrentOwner(__attribute__((unused)) bool ignoreOwner)
{
}

inline void SpinLock::assertUnowned()
{
}

inline void SpinLock::assertUnownedOrCurrentOwner(__attribute__((unused)) bool ignoreOwner)
{
}

inline void SpinLock::assertNotCurrentOwner()
{
}

inline void SpinLock::setOwner()
{
}

inline void SpinLock::unsetOwner()
{
}
#endif

#if USE_OLD_SPINLOCK_IMPLEMENTATION
	#ifdef __APPLE__
		#include "apple/SpinLockImplementation.hpp"
	#else
		#include "posix/SpinLockImplementation.hpp"
	#endif
#else
	#include "cxx/SpinLockImplementation.hpp"
#endif

#undef SPINLOCK_INTERNAL_TYPE


#endif // SPIN_LOCK_HPP
