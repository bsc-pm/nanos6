#ifndef APPLE_SPIN_LOCK_IMPLEMENTATION_HPP
#define APPLE_SPIN_LOCK_IMPLEMENTATION_HPP


#ifndef SPIN_LOCK_HPP
	#error Include SpinLock.h instead
#endif


inline SpinLock::SpinLock()
{
	pthread_spin_init(&_lock, PTHREAD_PROCESS_PRIVATE);
}

inline SpinLock::~SpinLock()
{
	pthread_spin_destroy(&_lock);
}

inline void SpinLock::lock()
{
	pthread_spin_lock(&_lock);
}

inline bool SpinLock::tryLock()
{
	return (pthread_spin_trylock(&_lock) == 0);
}

inline void SpinLock::unlock()
{
	pthread_spin_unlock(&_lock);
}


#endif // APPLE_SPIN_LOCK_IMPLEMENTATION_HPP
