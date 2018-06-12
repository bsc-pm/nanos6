/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef SPIN_LOCK_HPP
#define SPIN_LOCK_HPP

#include "SpinLockNoDebug.hpp"
#include "SpinLockOwnerDebug.hpp"

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



template <class DEBUG_KIND>
class CustomizableSpinLock: public DEBUG_KIND {
private:
	CustomizableSpinLock operator=(const CustomizableSpinLock &) = delete;
	CustomizableSpinLock(const CustomizableSpinLock & ) = delete;
	
	SPINLOCK_INTERNAL_TYPE _lock;
	
public:
	inline CustomizableSpinLock();
	inline ~CustomizableSpinLock();
	inline void lock();
	inline bool tryLock();
	inline void unlock(bool ignoreOwner = false);
#ifndef NDEBUG
	inline bool isLockedByThisThread();
#endif
};


#ifndef NDEBUG
using SpinLock = CustomizableSpinLock<SpinLockOwnerDebug>;
#else
using SpinLock = CustomizableSpinLock<SpinLockNoDebug>;
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
