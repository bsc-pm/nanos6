/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2018-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef SPIN_LOCK_HPP
#define SPIN_LOCK_HPP

#include <atomic>
#include <cassert>

#include "SpinWait.hpp"

class SpinLock {
private:
	SpinLock operator=(const SpinLock &) = delete;
	SpinLock(const SpinLock & ) = delete;

	std::atomic<bool> _lock;

public:
	static constexpr int ReadsBetweenCompareExchange = 1000;

	inline SpinLock() : _lock(false)
	{
	}

	inline ~SpinLock()
	{
		assert(!_lock.load());
	}

	inline void lock()
	{
		bool expected = false;
		while (!_lock.compare_exchange_weak(expected, true, std::memory_order_acquire)) {
			int spinsLeft = ReadsBetweenCompareExchange;
			do {
				spinWait();
				spinsLeft--;
			} while (_lock.load(std::memory_order_relaxed) && (spinsLeft > 0));

			spinWaitRelease();

			expected = false;
		}
	}

	inline bool tryLock()
	{
		bool expected = false;
		return _lock.compare_exchange_strong(expected, true, std::memory_order_acquire);
	}

	inline void unlock()
	{
		_lock.store(false, std::memory_order_release);
	}
};

#endif // SPIN_LOCK_HPP
