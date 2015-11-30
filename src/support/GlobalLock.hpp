#ifndef GLOBAL_LOCK_HPP
#define GLOBAL_LOCK_HPP


#include "lowlevel/SpinLock.hpp"


class GlobalLock {
private:
	static SpinLock _lock;
	
public:
	static void lock()
	{
		_lock.lock();
	}
	
	static void unlock()
	{
		_lock.unlock();
	}
};


#endif // GLOBAL_LOCK_HPP
