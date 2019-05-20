/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
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
	struct node {
		std::atomic<uint64_t> ticket;
		std::atomic<uint64_t> cpuIdx;
	};
	
protected:
	alignas(CACHELINE_SIZE) Padded<node> *_waitq;
	alignas(CACHELINE_SIZE) std::atomic<uint64_t> _head;
	alignas(CACHELINE_SIZE) uint64_t _next;
	alignas(CACHELINE_SIZE) uint64_t _size;
	
	inline void pushWaitingCPU(uint64_t const ticket, uint64_t const cpuIndex)
	{
		_waitq[ticket % _size].cpuIdx.store(ticket+cpuIndex, std::memory_order_relaxed);
	}
	
public:
	SubscriptionLock(size_t size)
		: _head(size), _next(size+1), _size(size)
	{
		_waitq = (Padded<node> *) MemoryAllocator::alloc(size * sizeof(Padded<node>));
		for (size_t i = 0; i < size; i++) {
			new (&_waitq[i]) Padded<node>();
		}
		
		_waitq[0].ticket = _size;
		_waitq[0].cpuIdx = 0;
	}
	
	~SubscriptionLock()
	{
		MemoryAllocator::free(_waitq, _size * sizeof(Padded<node>));
	}
	
	inline bool popWaitingCPU(uint64_t const myTicket, uint64_t & cpu)
	{
		uint64_t const cpuIdx = _waitq[myTicket % _size].cpuIdx.load(std::memory_order_relaxed);
		cpu = cpuIdx - myTicket;
		return  cpuIdx > myTicket;
	}
	
	inline void lock()
	{
		const uint64_t head = _head.fetch_add(1, std::memory_order_relaxed);
		const uint64_t idx = head % _size;
		while (_waitq[idx].ticket.load(std::memory_order_acquire) != head) {
			spinWait();
		}
	}
	
	inline uint64_t subscribeOrLock(uint64_t const cpuIndex)
	{
		const uint64_t head = _head.fetch_add(1, std::memory_order_relaxed);
		pushWaitingCPU(head, cpuIndex);
		const uint64_t idx = head % _size;
		while (_waitq[idx].ticket.load(std::memory_order_acquire) < head) {
			spinWait();
		}
		return head;
	}
	
	inline bool tryLock()
	{
		uint64_t head = _head.load(std::memory_order_relaxed);
		const uint64_t idx = head % _size;
		if (_waitq[idx].ticket.load(std::memory_order_relaxed) != head) {
			return false;
		}
		return std::atomic_compare_exchange_weak_explicit(
				&_head,
				&head,
				head+1,
				std::memory_order_acquire,
				std::memory_order_relaxed);
	}
	
	inline void unsubscribe()
	{
		const uint64_t idx = _next % _size;
		_waitq[idx].ticket.store(_next++, std::memory_order_release);
	}
	
	inline uint64_t waiting()
	{
		return _head.load(std::memory_order_relaxed) - (_next);
	}
};

#endif //SUBSCRIPTION_LOCK_HPP

