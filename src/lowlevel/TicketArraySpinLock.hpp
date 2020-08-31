/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TICKET_ARRAY_SPIN_LOCK_HPP
#define TICKET_ARRAY_SPIN_LOCK_HPP

#include <atomic>
#include <cstdint>

#include "MemoryAllocator.hpp"
#include "SpinWait.hpp"
#include "Padding.hpp"


class TicketArraySpinLock {
	typedef std::atomic<uint64_t> atomic_t;

	//! These are aligned on a cache line boundary in order to avoid false sharing
	alignas(CACHELINE_SIZE) Padded<atomic_t> *_buffer;
	alignas(CACHELINE_SIZE) atomic_t _head;
	alignas(CACHELINE_SIZE) uint64_t _next;
	alignas(CACHELINE_SIZE) uint64_t _size;

public:
	TicketArraySpinLock(size_t size) :
		_head(0),
		_next(0),
		_size(size)
	{
		_buffer = (Padded<atomic_t> *) MemoryAllocator::alloc(_size * sizeof(Padded<atomic_t>));
		for (size_t i = 0; i < _size; i++) {
			new (&_buffer[i]) Padded<atomic_t>(0);
		}
	}

	~TicketArraySpinLock()
	{
		MemoryAllocator::free(_buffer, _size * sizeof(Padded<atomic_t>));
	}

	inline void lock()
	{
		const uint64_t head = _head.fetch_add(1, std::memory_order_relaxed);
		const uint64_t idx = head % _size;
		while (_buffer[idx].load(std::memory_order_acquire) != head) {
			spinWait();
		}
	}

	inline bool tryLock()
	{
		uint64_t head = _head.load(std::memory_order_relaxed);
		const uint64_t idx = head % _size;
		if (_buffer[idx].load(std::memory_order_relaxed) != head)
			return false;

		return std::atomic_compare_exchange_strong_explicit(
			&_head, &head, head + 1,
			std::memory_order_acquire,
			std::memory_order_relaxed);
	}

	inline void unlock()
	{
		const uint64_t idx = ++_next % _size;
		_buffer[idx].store(_next, std::memory_order_release);
	}
};

#endif // TICKET_ARRAY_SPIN_LOCK_HPP
