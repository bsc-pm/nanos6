/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef ATOMIC_BITSET_HPP
#define ATOMIC_BITSET_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

template <typename backingstorage_t = uint64_t>
class AtomicBitset {
	typedef std::atomic<backingstorage_t> backing_t;
	size_t _size;
	std::vector<backing_t> _storage;

	static inline size_t getVecSize(size_t size)
	{
		// Round up
		return (size + sizeof(backingstorage_t) - 1) / sizeof(backingstorage_t);
	}

public:
	AtomicBitset(size_t size) :
		_size(size),
		_storage(getVecSize(size))
	{
		for (backing_t &elem : _storage)
			elem = 0;
	}

	void set(size_t pos)
	{
		assert(pos < _size);
		backing_t &elem = _storage[pos / sizeof(backingstorage_t)];
		elem.fetch_or(1 << (pos % sizeof(backingstorage_t)), std::memory_order_relaxed);
	}

	void reset(size_t pos)
	{
		assert(pos < _size);
		backing_t &elem = _storage[pos / sizeof(backingstorage_t)];
		elem.fetch_and(~(1 << (pos % sizeof(backingstorage_t))), std::memory_order_relaxed);
	}

	// Set the first zero bit and return the index. -1 if no zero bits were found.
	int setFirst()
	{
		size_t currentPos = 0;

		while (currentPos < _size) {
			backing_t &elem = _storage[currentPos / sizeof(backingstorage_t)];
			backingstorage_t value = elem.load(std::memory_order_relaxed);
			int firstOne = __builtin_ffsll(~value);
			while (firstOne && (currentPos + firstOne) <= _size) {
				// Found a bit to zero. Lets set it.
				assert(!(value & (1 << (firstOne - 1))));
				if (elem.compare_exchange_strong(value, value | (1 << (firstOne - 1))))
					return (currentPos + firstOne - 1);
				// Raced with another thread
				firstOne = __builtin_ffsll(~value);
			}

			currentPos += sizeof(backingstorage_t);
		}

		// We found no suitable position.
		return -1;
	}

	bool none()
	{
		for (size_t i = 0; i < getVecSize(_size); ++i) {
			if (_storage[i].load())
				return false;
		}

		return true;
	}
};

#endif // ATOMIC_BITSET_HPP