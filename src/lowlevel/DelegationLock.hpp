/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef DELEGATION_LOCK_HPP
#define DELEGATION_LOCK_HPP

#include <atomic>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

#include "MemoryAllocator.hpp"
#include "SpinWait.hpp"
#include "Padding.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


template <typename T>
class DelegationLock {
private:
	struct Node {
		std::atomic<uint64_t> _ticket;
		std::atomic<uint64_t> _cpuId;

		Node() :
			_ticket(0),
			_cpuId(0)
		{
		}
	};

	struct Item {
		uint64_t _ticket;
		T _item;
	};

#ifndef NDEBUG
	//! The amount of threads currently waiting in the lock
	std::atomic<size_t> _subscribedThreads;
#endif

protected:

	alignas(CACHELINE_SIZE) Padded<Node> *_waitQueue;
	alignas(CACHELINE_SIZE) Padded<Item> *_items;
	alignas(CACHELINE_SIZE) std::atomic<uint64_t> _head;
	alignas(CACHELINE_SIZE) uint64_t _size;
	alignas(CACHELINE_SIZE) uint64_t _next;

public:

	//! \brief Construct a delegation lock
	//!
	//! \param[in] size The number of slots in the delegation lock, i.e., the
	//! maximum number of threads that can access this lock simultaneously
	DelegationLock(size_t size) :
#ifndef NDEBUG
		_subscribedThreads(0),
#endif
		_head(size),
		_size(size),
		_next(size + 1)
	{
		_waitQueue = (Padded<Node> *) MemoryAllocator::alloc(size * sizeof(Padded<Node>));
		_items = (Padded<Item> *) MemoryAllocator::alloc(size * sizeof(Padded<Item>));
		assert(_waitQueue != nullptr);
		assert(_items != nullptr);

		for (size_t i = 0; i < size; i++) {
			new (&_waitQueue[i]) Padded<Node>();
			new (&_items[i]) Padded<Item>();
		}

		_waitQueue[0]._ticket = _size;
		_waitQueue[0]._cpuId = 0;
	}

	//! \breif Destroy a delegation lock
	~DelegationLock()
	{
#ifndef NDEBUG
		if (_subscribedThreads.load(std::memory_order_relaxed) > 0) {
			FatalErrorHandler::fail("Destroying a lock with threads inside");
		}
#endif
		MemoryAllocator::free(_waitQueue, _size * sizeof(Padded<Node>));
		MemoryAllocator::free(_items, _size * sizeof(Padded<Item>));
	}

	//! \brief Acquire the lock
	inline void lock()
	{
#ifndef NDEBUG
		size_t subscribedThreads = _subscribedThreads.fetch_add(1, std::memory_order_relaxed);
		if (subscribedThreads >= _size) {
			FatalErrorHandler::fail("Exceeded the maximum amount of threads inside the lock");
		}
#endif
		const uint64_t head = _head.fetch_add(1, std::memory_order_relaxed);
		const uint64_t id = head % _size;

		// Wait until it is our turn
		while (_waitQueue[id]._ticket.load(std::memory_order_acquire) != head) {
			spinWait();
		}
		spinWaitRelease();
	}

	//! \brief Acquire the lock or wait until someone serves an item
	//!
	//! This function blocks the current compute place (busy waiting)
	//! until we acquire the lock or any other compute places has
	//! served us an item (delegation)
	//!
	//! \param[in] cpuIndex The index of the CPU
	//! \param[out] item The served item if we did not get the lock
	//!
	//! \return Whether the lock was acquired
	inline bool lockOrDelegate(uint64_t const cpuIndex, T &item)
	{
#ifndef NDEBUG
		size_t subscribedThreads = _subscribedThreads.fetch_add(1, std::memory_order_relaxed);
		if (subscribedThreads >= _size) {
			FatalErrorHandler::fail("Exceeded the maximum amount of threads inside the lock");
		}
#endif
		const uint64_t head = _head.fetch_add(1, std::memory_order_relaxed);
		const uint64_t id = head % _size;
		_waitQueue[id]._cpuId.store(head + cpuIndex, std::memory_order_relaxed);

		// Wait until it is our turn or someone else has served us an item
		while (_waitQueue[id]._ticket.load(std::memory_order_acquire) < head) {
			spinWait();
		}
		spinWaitRelease();

		if (_items[cpuIndex]._ticket != head) {
			// We acquired the lock
			return true;
		}

		// Save the item they served us
		item = _items[cpuIndex]._item;

		return false;
	}

	//! \brief Try to acquire the lock
	//!
	//! \return Whether the lock was acquired
	inline bool tryLock()
	{
		uint64_t head = _head.load(std::memory_order_relaxed);
		const uint64_t id = head % _size;
		if (_waitQueue[id]._ticket.load(std::memory_order_relaxed) != head) {
			return false;
		}

		bool success = std::atomic_compare_exchange_weak_explicit(
			&_head,
			&head,
			head + 1,
			std::memory_order_acquire,
			std::memory_order_relaxed
		);
#ifndef NDEBUG
		if (success) {
			size_t subscribedThreads = _subscribedThreads.fetch_add(1, std::memory_order_relaxed);
			if (subscribedThreads >= _size) {
				FatalErrorHandler::fail("Exceeded the maximum amount of threads inside the lock");
			}
		}
#endif
		return success;
	}

	//! \brief Release the lock
	//!
	//! This function must be called with the lock acquired
	inline void unlock()
	{
		popFront();
	}

	//! \brief Check whether there are no compute places waiting
	//!
	//! This function must be called with the lock acquired
	//!
	//! \return Whether there are no compute places waiting
	inline bool empty() const
	{
		const uint64_t cpuId = _waitQueue[_next % _size]._cpuId.load(std::memory_order_relaxed);
		return (cpuId < _next);
	}

	//! \brief Get the index of the first waiting compute place
	//!
	//! This function must be called with the lock acquired and should be
	//! a waiting compute place. Use the empty() function to check if there
	//! are compute places waiting
	//!
	//! \return The index of the first waiting compute place
	inline uint64_t front() const
	{
		const uint64_t cpuId = _waitQueue[_next % _size]._cpuId.load(std::memory_order_relaxed);
		return (cpuId - _next);
	}

	//! \brief Unblock the first waiting compute place
	//!
	//! This function unblocks the first waiting compute place, which
	//! should have been served using the setItem() function, and then,
	//! it moves to the next waiting compute place (if any). This function
	//! must be called with the lock acquired and should be a waiting
	//! compute place
	inline void popFront()
	{
#ifndef NDEBUG
		assert(_subscribedThreads > 0);
		--_subscribedThreads;
#endif
		const uint64_t id = _next % _size;
		_waitQueue[id]._ticket.store(_next++, std::memory_order_release);
	}

	//! \brief Serve an item to a waiting compute place
	//!
	//! This function must be called with the lock acquired. A compute
	//! place can only be served a single time while is waiting
	//!
	//! \param[in] cpuIndex The index of the waiting compute place
	//! \param[in] item The item to ber served
	inline void setItem(const uint64_t cpuIndex, T item)
	{
		_items[cpuIndex]._item = item;
		_items[cpuIndex]._ticket = _next;
	}
};

#endif // DELEGATION_LOCK_HPP

