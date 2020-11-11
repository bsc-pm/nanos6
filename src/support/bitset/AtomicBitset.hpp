/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef ATOMIC_BITSET_HPP
#define ATOMIC_BITSET_HPP

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#define bitsizeof(t) (sizeof(t) * 8)

//! \brief Represents a bitset that can be modified concurrently by multiple threads
//!
//! AtomicBitset provides a std::bitset-like class that can be safely modified by multiple
//! threads by providing atomic set and reset operations.
//! It also provides a setFirst function to set the first zero bit, which has some weaker guarantees.
template <typename backingstorage_t = uint64_t>
class AtomicBitset {
	typedef std::atomic<backingstorage_t> backing_t;
	size_t _size;
	std::vector<backing_t> _storage;

	//! This looks a bit silly, but is needed because if we just use "1"
	//! the C standard says it's a 32-bit integer, and shifting it by more
	//! than 32 bits is undefined behaviour
	static const backingstorage_t ONE = 1;

	//! \brief Calculate the number of backing storage elements needed to hold a number of bits
	//!
	//! \param[in] size the number of bits to store
	//!
	//! \return the minimum number of backingstorage_t elements needed to hold "size" bits
	static inline size_t getVecSize(size_t size)
	{
		// Round up
		return (size + bitsizeof(backingstorage_t) - 1) / bitsizeof(backingstorage_t);
	}

	//! \brief Get a reference to the backing storage element that holds one specific bit
	//!
	//! \param[in] pos the position (index) of the searched bit
	//!
	//! \return a reference to the element storing this bit
	inline backing_t &getStorage(size_t pos)
	{
		assert(pos < _size);
		return _storage[pos / bitsizeof(backingstorage_t)];
	}

	//! \brief Get the index where a bit is located inside a backing storage element
	//!
	//! \param[in] pos the positon (index) of the searched bit inside the whole bitset
	//!
	//! \return the position (index) of the bit inside a single element
	static inline size_t getBitIndex(size_t pos)
	{
		return pos % bitsizeof(backingstorage_t);
	}

public:
	//! \brief Construct an AtomicBitset that can hold a minimum of bits
	//!
	//! \param[in] size minimum number of bits that the AtomicBitset must hold
	AtomicBitset(size_t size) :
		_size(size),
		_storage(getVecSize(size))
	{
		for (backing_t &elem : _storage)
			elem = 0;
	}

	//! \brief Set a single bit in the AtomicBitset (to 1)
	//!
	//! \remark This function is atomic but does not impose any ordering
	//! \remark This function is wait-free and has O(1) complexity
	//!
	//! \param[in] pos position of the bit to set
	inline void set(size_t pos)
	{
		backing_t &elem = getStorage(pos);
		//! Setting a bit is equivalent to the or with the single bit that has to be set
		elem.fetch_or(ONE << getBitIndex(pos), std::memory_order_relaxed);
	}

	//! \brief Reset a single bit in the AtomicBitset (to 0)
	//!
	//! \remark This function is atomic but does not impose any ordering
	//! \remark This function is wait-free and has O(1) complexity
	//!
	//! \param[in] pos position of the bit to reset
	inline void reset(size_t pos)
	{
		backing_t &elem = getStorage(pos);
		//! Resetting a bit is equal to the and with a mask that is all 1's except the position
		//! that we want to reset. We shift a one just like on the "set" and then bitwise negation (~)
		elem.fetch_and(~(ONE << getBitIndex(pos)), std::memory_order_relaxed);
	}

	//! \brief Set the first found zero-bit in the AtomicBitset
	//!
	//! \remark This function only provides one guarantee: if a position != -1 is
	//!         returned, that position has been atomically set and was a 0.
	//!			Note that it can still race with another thread and return -1 even if
	//!			there are bits reset to zero, or set a bit which is not the first anymore.
	//! \remark This function is lock-free and has O(size) complexity
	//!
	//! \return Index of the zero-bit that has been set, or -1 if no zero-bit was found.
	inline int setFirst()
	{
		size_t currentPos = 0;

		while (currentPos < _size) {
			backing_t &elem = getStorage(currentPos);
			backingstorage_t value = elem.load(std::memory_order_relaxed);
			int firstOne = __builtin_ffsll(~value);

			while (firstOne && (currentPos + firstOne) <= _size) {
				//! Found a bit to zero. Lets set it.
				backingstorage_t mask = (ONE << (firstOne - 1));
				assert(!(value & mask));
				if (elem.compare_exchange_strong(value, value | mask, std::memory_order_relaxed))
					return (currentPos + firstOne - 1);
				//! Raced with another thread. This retry makes this function lock-free (not wait-free).
				firstOne = __builtin_ffsll(~value);
			}

			currentPos += bitsizeof(backingstorage_t);
		}

		//! We found no suitable position.
		return -1;
	}

	//! \brief Finds if there are no set bits in the AtomicBitset
	//!
	//! \remark This function provides no atomicity or ordering guarantees
	//! \remark This function is wait-free and has O(size) complexity
	//!
	//! \return true if no set bits are found, false otherwise
	inline bool none() const
	{
		for (size_t i = 0; i < getVecSize(_size); ++i) {
			if (_storage[i].load(std::memory_order_relaxed))
				return false;
		}

		return true;
	}
};

#endif // ATOMIC_BITSET_HPP
