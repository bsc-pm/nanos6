/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef SPIN_LOCK_DEADLOCK_DEBUG_HPP
#define SPIN_LOCK_DEADLOCK_DEBUG_HPP

#include <lowlevel/BacktraceRecording.hpp>
#include <lowlevel/SpinLock.hpp>
#include <lowlevel/SpinLockNoDebug.hpp>

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>


class WorkerThread;
namespace ompss_debug {
	__attribute__((weak)) WorkerThread *getCurrentWorkerThread();
}


class SpinLockDeadlockDebug {
private:
	WorkerThread *_owner;
	RecordedBacktrace _lastStatusChangeBacktrace;
	
	typedef CustomizableSpinLock<SpinLockNoDebug> aux_lock_t;
	
	inline void lockOutput()
	{
		static aux_lock_t lock;
		lock.lock();
	}
	
	struct DeadlockCheckInfo {
		typedef std::map<WorkerThread *, std::map<SpinLockDeadlockDebug *, RecordedBacktrace>> map_t;
		
		aux_lock_t _lock;
		map_t _lockingMap;
	};
	
	inline DeadlockCheckInfo &getDeadlockCheckInfo()
	{
		static DeadlockCheckInfo deadlockCheckInfo;
		return deadlockCheckInfo;
	}
	
	
public:
	inline SpinLockDeadlockDebug()
		: _owner(nullptr)
	{
	}
	
	
	inline void willLock()
	{
		WorkerThread *thisThread = ompss_debug::getCurrentWorkerThread();
		
		RecordedBacktrace currentBacktrace;
		currentBacktrace.capture(0);
		
		DeadlockCheckInfo &deadlockCheckInfo = getDeadlockCheckInfo();
		deadlockCheckInfo._lock.lock();
		
		auto &ourLocks = deadlockCheckInfo._lockingMap[thisThread];
		ourLocks[this] = std::move(currentBacktrace);
		
		WorkerThread *owner = _owner;
		if (owner != nullptr) {
			auto &ownerLocks = deadlockCheckInfo._lockingMap[owner];
			
			for (auto &lockOwnedByThisThread : ourLocks) {
				if (lockOwnedByThisThread.first == this) {
					continue;
				}
				
				auto lockOwnedByCurrentOwnerIt = ownerLocks.find(lockOwnedByThisThread.first);
				if (lockOwnedByCurrentOwnerIt != ownerLocks.end()) {
					auto &lockOwnedByCurrentOwner = *lockOwnedByCurrentOwnerIt;
					
					lockOutput();
					
					std::cerr << "Deadlock detected between worker threads " << thisThread << " and " << owner << std::endl;
					std::cerr << "This thread " << thisThread << " owns " << lockOwnedByThisThread.first << " since:" << std::endl;
					std::cerr << lockOwnedByThisThread.second << std::endl;
					std::cerr << "Which thread " << owner << " is trying to get from:" << std::endl;
					std::cerr << lockOwnedByCurrentOwner.second << std::endl;
					std::cerr << std::endl;
					std::cerr << "Thread " << owner << " already owns " << this << " since:" << std::endl;
					std::cerr << ownerLocks[this] << std::endl;
					std::cerr << "Which this thread " << thisThread << " is also trying to lock from:" << std::endl;
					std::cerr << ourLocks[this] << std::endl;
					
					abort();
				}
			}
		}
		
		deadlockCheckInfo._lock.unlock();
	}
	
	
	inline void assertCurrentOwner(__attribute__((unused)) bool ignoreOwner)
	{
		if (!ignoreOwner && (_owner != ompss_debug::getCurrentWorkerThread())) {
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
		if (!ignoreOwner && (_owner != nullptr) && (_owner != ompss_debug::getCurrentWorkerThread())) {
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
		if (_owner == ompss_debug::getCurrentWorkerThread()) {
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
		WorkerThread *thisThread = ompss_debug::getCurrentWorkerThread();
		
		DeadlockCheckInfo &deadlockCheckInfo = getDeadlockCheckInfo();
		deadlockCheckInfo._lock.lock();
		
		_owner = thisThread;
		_lastStatusChangeBacktrace.capture(0);
		
		auto &ourLocks = deadlockCheckInfo._lockingMap[thisThread];
		if (ourLocks.find(this) == ourLocks.end()) {
			// A successful tryLock
			RecordedBacktrace currentBacktrace;
			currentBacktrace.capture(0);
			
			ourLocks[this] = std::move(currentBacktrace);
		}
		
		deadlockCheckInfo._lock.unlock();
	}
	
	
	inline void unsetOwner()
	{
		WorkerThread *thisThread = ompss_debug::getCurrentWorkerThread();
		
		DeadlockCheckInfo &deadlockCheckInfo = getDeadlockCheckInfo();
		deadlockCheckInfo._lock.lock();
		
		_owner = nullptr;
		_lastStatusChangeBacktrace.capture(0);
		
		auto &ourLocks = deadlockCheckInfo._lockingMap[thisThread];
		ourLocks.erase(this);
		
		deadlockCheckInfo._lock.unlock();
	}
	
	
	inline bool isLockedByThisThread()
	{
		return (_owner == ompss_debug::getCurrentWorkerThread());
	}
	
};


#endif // SPIN_LOCK_DEADLOCK_DEBUG_HPP
