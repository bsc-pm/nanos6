/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/


#ifndef TICKET_ARRAY_SPIN_LOCK_HPP
#define TICKET_ARRAY_SPIN_LOCK_HPP

#include <atomic>

#include "MemoryAllocator.hpp"
#include "SpinWait.hpp"
#include "Padding.hpp"

class TicketArraySpinLock {
	// These are aligned on a cache line boundary in order to avoid false sharing:
	alignas(CACHELINE_SIZE) Padded<std::atomic_size_t> *_buffer;
	alignas(CACHELINE_SIZE) std::atomic_size_t _head;
	alignas(CACHELINE_SIZE) size_t _next;
	alignas(CACHELINE_SIZE) size_t _size;

public:
	TicketArraySpinLock(size_t size)
		: _head(0), _next(0), _size(size)
	{
		_buffer = (Padded<std::atomic_size_t> *) MemoryAllocator::alloc(_size * sizeof(Padded<std::atomic_size_t>));
		for (size_t i = 0; i < _size; i++) {
			new (&_buffer[i]) Padded<std::atomic_size_t>();
		}
	}

	~TicketArraySpinLock()
	{
		MemoryAllocator::free(_buffer, _size * sizeof(Padded<std::atomic_size_t>));
	}

	inline bool tryLock()
	{
		const size_t head = _head.fetch_add(1, std::memory_order_relaxed);
		const size_t idx = head % _size;
		return !(_buffer[idx].load(std::memory_order_acquire) != head);
	}

	inline void lock()
	{
		const size_t head = _head.fetch_add(1, std::memory_order_relaxed);
		const size_t idx = head % _size;
		while (_buffer[idx].load(std::memory_order_acquire) != head) {
			spinWait();
		}
		_next = (head+1);
	}

	inline void unlock()
	{
		const size_t idx = _next % _size;
		_buffer[idx].store(_next, std::memory_order_release);
	}
};

#endif // TICKET_ARRAY_SPIN_LOCK_HPP
