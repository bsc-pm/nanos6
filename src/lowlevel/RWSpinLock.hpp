/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef RW_SPIN_LOCK_HPP
#define RW_SPIN_LOCK_HPP


#include <atomic>
#include <cassert>

// Meets SharedLockable, Lockable and BasicLockable requirements
class RWSpinLock {
private:
	std::atomic<int> _lock;

public:
	RWSpinLock() :
		_lock(0)
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

	inline bool readTryLock()
	{
		int value = _lock.load(std::memory_order_relaxed);
		if (value < 0)
			return false;

		return _lock.compare_exchange_weak(value, value + 1,
			std::memory_order_acquire,
			std::memory_order_relaxed);
	}

	inline void readUnlock()
	{
		__attribute__((unused)) int value = _lock.fetch_sub(1, std::memory_order_release);
		assert(value > 0);
	}

	inline bool writeTryLock()
	{
		int value = _lock.load(std::memory_order_relaxed);
		if (value != 0)
			return false;

		return _lock.compare_exchange_weak(value, -1,
			std::memory_order_acquire,
			std::memory_order_relaxed);
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
		int expected = -1;
		bool successful = _lock.compare_exchange_strong(expected, 0, std::memory_order_release);
		assert(successful);
#else
		_lock.store(0, std::memory_order_release);
#endif
	}

	// SharedLockable requirements for std::shared_lock
	inline void lock_shared()
	{
		readLock();
	}

	inline bool try_lock_shared()
	{
		return readTryLock();
	}

	inline void unlock_shared()
	{
		readUnlock();
	}

	// Lockable requirements for std::unique_lock
	inline void lock()
	{
		writeLock();
	}

	inline bool try_lock()
	{
		return writeTryLock();
	}

	inline void unlock()
	{
		writeUnlock();
	}
};


#endif // RW_SPIN_LOCK_HPP
