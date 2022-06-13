/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2022 Barcelona Supercomputing Center (BSC)
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

	//! Keep these fields on the same cacheline since they are not modified
	alignas(CACHELINE_SIZE) Padded<atomic_t> *_buffer;
	const uint64_t _size;

	//! Keep these fields occupying two cachelines to prevent false sharing and
	//! undesired conflicts due to prefetching
	alignas(CACHELINE_SIZE * 2) atomic_t _head;
	alignas(CACHELINE_SIZE * 2) uint64_t _next;

public:
	TicketArraySpinLock(size_t size) :
		_buffer(nullptr),
		_size(size),
		_head(0),
		_next(0)
	{
		_buffer = (Padded<atomic_t> *) MemoryAllocator::allocAligned(_size * sizeof(Padded<atomic_t>));
		for (size_t i = 0; i < _size; i++) {
			new (&_buffer[i]) Padded<atomic_t>(0);
		}
	}

	~TicketArraySpinLock()
	{
		MemoryAllocator::freeAligned(_buffer, _size * sizeof(Padded<atomic_t>));
	}

	inline void lock()
	{
		const uint64_t head = _head.fetch_add(1, std::memory_order_relaxed);
		const uint64_t idx = head % _size;
		while (_buffer[idx].load(std::memory_order_relaxed) != head) {
			spinWait();
		}
		spinWaitRelease();

		std::atomic_thread_fence(std::memory_order_acquire);
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
