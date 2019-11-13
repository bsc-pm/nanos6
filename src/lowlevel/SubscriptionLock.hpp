/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef SUBSCRIPTION_LOCK_HPP
#define SUBSCRIPTION_LOCK_HPP

#include <atomic>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

#include "MemoryAllocator.hpp"
#include "SpinWait.hpp"


class SubscriptionLock {

	struct Node {
		std::atomic<uint64_t> _ticket;
		std::atomic<uint64_t> _cpuId;
	};

protected:

	alignas(CACHELINE_SIZE) Padded<Node> *_waitQueue;
	alignas(CACHELINE_SIZE) std::atomic<uint64_t> _head;
	alignas(CACHELINE_SIZE) uint64_t _next;
	alignas(CACHELINE_SIZE) uint64_t _size;

public:

	//! \brief SubscriptionLock's constructor
	//!
	//! \param[in] size The number of slots in the subscription lock, i.e., the
	//! maximum number of threads that can access this lock simultaneously
	SubscriptionLock(size_t size)
		: _head(size), _next(size+1), _size(size)
	{
		_waitQueue = (Padded<Node> *) MemoryAllocator::alloc(size * sizeof(Padded<Node>));
		for (size_t i = 0; i < size; i++) {
			new (&_waitQueue[i]) Padded<Node>();
		}

		_waitQueue[0]._ticket = _size;
		_waitQueue[0]._cpuId = 0;
	}

	~SubscriptionLock()
	{
		MemoryAllocator::free(_waitQueue, _size * sizeof(Padded<Node>));
	}

	//! \brief Try to get a currently subscribed CPU
	//!
	//! \param[in] myTicket The ticket related to the subscriber (if it exists)
	//! \param[out] cpuId Upon returning true, this parameter contains the id
	//! of the subscriber CPU we've just popped from the waiting queue
	//!
	//! \return Whether a CPU was popped from the waiting queue
	inline bool popWaitingCPU(uint64_t const myTicket, uint64_t &cpuId)
	{
		uint64_t const innerCPUId = _waitQueue[myTicket % _size]._cpuId.load(std::memory_order_relaxed);
		cpuId = innerCPUId - myTicket;
		return innerCPUId > myTicket;
	}

	//! \brief Obtain the lock without subscribing
	inline void lock()
	{
		const uint64_t head = _head.fetch_add(1, std::memory_order_relaxed);
		const uint64_t id = head % _size;

		// Wait until it's our turn
		while (_waitQueue[id]._ticket.load(std::memory_order_acquire) != head) {
			spinWait();
		}
	}

	//! \brief Subscribe or obtain the lock
	//!
	//! \param[in] cpuIndex The index of the CPU
	//!
	//! \return The ticket obtained for the current subscription
	inline uint64_t subscribeOrLock(uint64_t const cpuIndex)
	{
		const uint64_t head = _head.fetch_add(1, std::memory_order_relaxed);
		const uint64_t id = head % _size;
		_waitQueue[id]._cpuId.store(head+cpuIndex, std::memory_order_relaxed);

		// Wait until it's our turn or someone else has served our subscription
		while (_waitQueue[id]._ticket.load(std::memory_order_acquire) < head) {
			spinWait();
		}

		return head;
	}

	//! \brief Try to obtain the lock
	//!
	//! \return Whether the lock was obtained
	inline bool tryLock()
	{
		uint64_t head = _head.load(std::memory_order_relaxed);
		const uint64_t id = head % _size;
		if (_waitQueue[id]._ticket.load(std::memory_order_relaxed) != head) {
			return false;
		}

		return std::atomic_compare_exchange_weak_explicit(
			&_head,
			&head,
			head+1,
			std::memory_order_acquire,
			std::memory_order_relaxed
		);
	}

	//! \brief Unsubscribe from the lock (advance the ticket)
	inline void unsubscribe()
	{
		const uint64_t id = _next % _size;
		_waitQueue[id]._ticket.store(_next++, std::memory_order_release);
	}

	inline uint64_t waiting()
	{
		return _head.load(std::memory_order_relaxed) - (_next);
	}
};

#endif // SUBSCRIPTION_LOCK_HPP

