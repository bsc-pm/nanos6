/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef RW_SPIN_LOCK_HPP
#define RW_SPIN_LOCK_HPP



#include <atomic>
#include <cassert>

class RWSpinLock {
private:
	std::atomic<int> _lock;

public:
	RWSpinLock()
		: _lock(0)
	{
	}

	inline void readLock()
	{
		bool successful = false;
		while (!successful) {
			int value;
			while ((value = _lock.load(std::memory_order_relaxed)) < 0);

			successful = _lock.compare_exchange_weak(value, value + 1,
					std::memory_order_acquire,
					std::memory_order_relaxed);
		}
	}

	inline void readUnlock()
	{
		int value = _lock.fetch_sub(1, std::memory_order_release);
		assert(value > 0);
	}

	inline void writeLock()
	{
		bool successful = false;
		while (!successful) {
			int value;
			while ((value = _lock.load(std::memory_order_relaxed)) != 0);

			successful = _lock.compare_exchange_weak(value, -1,
					std::memory_order_acquire,
					std::memory_order_relaxed);
		}
	}

	inline void writeUnlock()
	{
#ifndef NDEBUG
		__attribute__((unused)) bool successful;
		int expected = -1;
		successful = _lock.compare_exchange_strong(expected, 0,
				std::memory_order_release);
		assert(successful);
#else
		_lock.store(0, std::memory_order_release);
#endif
	}

};


#endif // RW_SPIN_LOCK_HPP
