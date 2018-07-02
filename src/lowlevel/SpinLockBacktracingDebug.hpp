/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef SPIN_LOCK_BACKTRACING_DEBUG_HPP
#define SPIN_LOCK_BACKTRACING_DEBUG_HPP

#include <lowlevel/BacktraceRecording.hpp>
#include <lowlevel/SpinLock.hpp>
#include <lowlevel/SpinLockNoDebug.hpp>

#include <cassert>
#include <cstdlib>
#include <iostream>


namespace ompss_debug {
	void *getCurrentThread();
}


class SpinLockBacktracingDebug {
private:
	void *_owner;
	RecordedBacktrace _lastStatusChangeBacktrace;
	
	inline void lockOutput()
	{
		static CustomizableSpinLock<SpinLockNoDebug> lock;
		lock.lock();
	}
	
	
public:
	inline SpinLockBacktracingDebug()
		: _owner(nullptr)
	{
	}
	
	inline void willLock()
	{
	}
	
	inline void assertCurrentOwner(__attribute__((unused)) bool ignoreOwner)
	{
		if (!ignoreOwner && (_owner != ompss_debug::getCurrentThread()) && (ompss_debug::getCurrentThread() != nullptr)) {
			lockOutput();
			
			RecordedBacktrace currentBacktrace;
			currentBacktrace.capture(0);
			
			std::cerr << "Assertion failure: expected SpinLock to be locked by the current thread at:" << std::endl;
			std::cerr << currentBacktrace << std::endl;
			std::cerr << "Previous status change at:" << std::endl;
			std::cerr << _lastStatusChangeBacktrace;
			
			abort();
		}
	}
	
	inline void assertUnowned()
	{
		if (_owner != nullptr) {
			lockOutput();
			
			RecordedBacktrace currentBacktrace;
			currentBacktrace.capture(0);
			
			std::cerr << "Assertion failure: expected SpinLock to be unlocked at:" << std::endl;
			std::cerr << currentBacktrace << std::endl;
			std::cerr << "Previous status change at:" << std::endl;
			std::cerr << _lastStatusChangeBacktrace;
			
			abort();
		}
	}
	
	inline void assertUnownedOrCurrentOwner(__attribute__((unused)) bool ignoreOwner)
	{
		if (!ignoreOwner && (_owner != nullptr) && (_owner != ompss_debug::getCurrentThread())) {
			lockOutput();
			
			RecordedBacktrace currentBacktrace;
			currentBacktrace.capture(0);
			
			std::cerr << "Assertion failure: expected SpinLock to be either unlocked on owned by the current thread at:" << std::endl;
			std::cerr << currentBacktrace << std::endl;
			std::cerr << "Previous status change at:" << std::endl;
			std::cerr << _lastStatusChangeBacktrace;
			
			abort();
		}
	}
	
	inline void assertNotCurrentOwner()
	{
		if ((_owner == ompss_debug::getCurrentThread()) && (ompss_debug::getCurrentThread() != nullptr)) {
			lockOutput();
			
			RecordedBacktrace currentBacktrace;
			currentBacktrace.capture(0);
			
			std::cerr << "Assertion failure: expected SpinLock to be not be owned by the current thread at:" << std::endl;
			std::cerr << currentBacktrace << std::endl;
			std::cerr << "Previous status change at:" << std::endl;
			std::cerr << _lastStatusChangeBacktrace;
			
			abort();
		}
	}
	
	inline void setOwner()
	{
		_owner = ompss_debug::getCurrentThread();
		if (_owner != nullptr) {
			_lastStatusChangeBacktrace.capture(0);
		}
	}
	
	inline void unsetOwner()
	{
		if (_owner != nullptr) {
			_owner = nullptr;
			_lastStatusChangeBacktrace.capture(0);
		}
	}
	
	inline bool isLockedByThisThread()
	{
		return (_owner == ompss_debug::getCurrentThread());
	}
	
};


#endif // SPIN_LOCK_BACKTRACING_DEBUG_HPP
