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
	
	inline void assertCurrentOwner();
	inline void assertUnowned();
	inline void assertUnownedOrCurrentOwner();
	inline void setOwner();
	inline void unsetOwner();
	
public:
	inline SpinLock();
	inline ~SpinLock();
	inline void lock();
	inline bool tryLock();
	inline void unlock();
	inline bool isLockedByThisThread();
};


#ifndef NDEBUG
inline void SpinLock::assertCurrentOwner()
{
	assert(_owner == ompss_debug::getCurrentWorkerThread());
}

inline void SpinLock::assertUnowned()
{
	assert(_owner == nullptr);
}

inline void SpinLock::assertUnownedOrCurrentOwner()
{
	assert( (_owner == nullptr) || (_owner == ompss_debug::getCurrentWorkerThread()) ) ;
}

inline void SpinLock::setOwner()
{
	_owner = ompss_debug::getCurrentWorkerThread();
}

inline void SpinLock::unsetOwner()
{
	_owner = nullptr;
}
#else
inline void SpinLock::assertCurrentOwner()
{
}

inline void SpinLock::assertUnowned()
{
}

inline void SpinLock::assertUnownedOrCurrentOwner()
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
