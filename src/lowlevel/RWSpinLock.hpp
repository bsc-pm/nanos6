/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef RW_SPIN_LOCK_HPP
#define RW_SPIN_LOCK_HPP



#include "SpinLock.hpp"


class RWSpinLock {
private:
	SpinLock _readersSpinLock;
	long _readers;
	SpinLock _writerSpinLock;
	
public:
	RWSpinLock()
		: _readersSpinLock(), _readers(0), _writerSpinLock()
		{
		}
		
	inline void readLock()
	{
		_readersSpinLock.lock();
		if (++_readers == 1) {
			_writerSpinLock.lock();
		}
		_readersSpinLock.unlock();
	}
	
	inline void readUnlock()
	{
		_readersSpinLock.lock();
		if (--_readers == 0) {
			_writerSpinLock.unlock(/* Ignore the owner */ true);
		}
		_readersSpinLock.unlock();
	}
	
	inline void writeLock()
	{
		_writerSpinLock.lock();
	}
	
	inline void writeUnlock()
	{
		_writerSpinLock.unlock();
	}
	
#ifndef NDEBUG
	inline bool isWriteLockedByThisThread()
	{
		return _writerSpinLock.isLockedByThisThread();
	}
#endif
	
};


#endif // RW_SPIN_LOCK_HPP
