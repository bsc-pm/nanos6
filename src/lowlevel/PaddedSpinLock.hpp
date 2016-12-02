#ifndef PADDED_SPIN_LOCK_HPP
#define PADDED_SPIN_LOCK_HPP


#include "SpinLock.hpp"


template <int PADDING = 64>
class PaddedSpinLock {
private:
	char _frontPadding[PADDING - sizeof(SpinLock)];
	SpinLock _lock;
	char _backPadding[PADDING - sizeof(SpinLock)];
	
	template <int PADDING2>
	PaddedSpinLock operator=(const PaddedSpinLock<PADDING2> &) = delete;
	
	template <int PADDING2>
	PaddedSpinLock(const PaddedSpinLock<PADDING2> &) = delete;
	
public:
	inline PaddedSpinLock()
	{
	}
	
	inline void lock()
	{
		_lock.lock();
	}
	
	inline bool tryLock()
	{
		return _lock.tryLock();
	}
	
	inline void unlock()
	{
		_lock.unlock();
	}
	
	inline bool isLockedByThisThread()
	{
		return _lock.isLockedByThisThread();
	}
	
	inline SpinLock &getSpinLock()
	{
		return _lock;
	}
};


#endif // PADDED_SPIN_LOCK_HPP
