/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CONCURRENT_UNORDERED_LIST_HPP
#define CONCURRENT_UNORDERED_LIST_HPP


#include <atomic>
#include <cassert>
#include <queue>

#include <lowlevel/SpinLock.hpp>


template <typename T, int MAX_SLOTS = 4096, int MAX_CONCURRENCY_PER_SLOT = 2>
class ConcurrentUnorderedList;


class ConcurrentUnorderedListSlotManager {
protected:
	std::atomic<int> _nextFreeSlot;
	
	template <typename T, int MAX_SLOTS, int MAX_CONCURRENCY_PER_SLOT>
	friend class ConcurrentUnorderedList;
	
public:
	class Slot {
	private:
		int _index;
		
	public:
		Slot()
			: _index(-1)
		{
		}
		
		Slot(int index)
			: _index(index)
		{
		}
		
		operator int()
		{
			return _index;
		}
	};
	
	
	Slot getSlot()
	{
		int slot = _nextFreeSlot++;
		return Slot(slot);
	}
};


template <typename T, int MAX_SLOTS, int MAX_CONCURRENCY_PER_SLOT>
class ConcurrentUnorderedList {
private:
	struct SlotIndexContents {
		SpinLock _lock;
		std::queue<T> _queue;
	};
	
	struct SlotContents {
		std::atomic<int> _nextPosition;
		SlotIndexContents _queues[MAX_CONCURRENCY_PER_SLOT];
		
		SlotContents()
			: _nextPosition(0)
		{
		}
	};
	
	ConcurrentUnorderedListSlotManager &_slotManager;
	SlotContents _slots[MAX_SLOTS];
	
	
public:
	typedef ConcurrentUnorderedListSlotManager::Slot Slot;
	
	
	ConcurrentUnorderedList(ConcurrentUnorderedListSlotManager &slotManager)
		: _slotManager(slotManager), _slots()
	{
	}
	
	
	void push(T const &value, Slot &slot)
	{
		assert(slot != Slot());
		SlotContents &slotContents = _slots[slot];
		
		while (true) {
			int queueIndex = slotContents._nextPosition++;
			queueIndex = queueIndex % MAX_CONCURRENCY_PER_SLOT;
			
			if (slotContents._queues[queueIndex]._lock.tryLock()) {
				
				slotContents._queues[queueIndex]._queue.push(value);
				
				slotContents._queues[queueIndex]._lock.unlock();
				break;
			}
		}
	}
	
	
	bool pop(T &value, Slot &slot) __attribute__((warn_unused_result))
	{
		assert(slot != Slot());
		SlotContents &slotContents = _slots[slot];
		
		int remaining = MAX_CONCURRENCY_PER_SLOT;
		bool done[MAX_CONCURRENCY_PER_SLOT];
		
		for (int i = 0; i < MAX_CONCURRENCY_PER_SLOT; i++) {
			done[i] = false;
		}
		
		while (remaining > 0) {
			int queueIndex = slotContents._nextPosition++;
			queueIndex = queueIndex % MAX_CONCURRENCY_PER_SLOT;
			
			if (!done[queueIndex] && slotContents._queues[queueIndex]._lock.tryLock()) {
				if (!slotContents._queues[queueIndex]._queue.empty()) {
					value = slotContents._queues[queueIndex]._queue.front();
					slotContents._queues[queueIndex]._queue.pop();
					
					slotContents._queues[queueIndex]._lock.unlock();
					
					return true;
				} else {
					slotContents._queues[queueIndex]._lock.unlock();
					
					remaining--;
					done[queueIndex] = true;
				}
			}
		}
		
		return false;
	}
	
	
	bool weak_pop(T &value, Slot &slot) __attribute__((warn_unused_result))
	{
		assert(slot != Slot());
		SlotContents &slotContents = _slots[slot];
		
		int queueIndex = slotContents._nextPosition++;
		queueIndex = queueIndex % MAX_CONCURRENCY_PER_SLOT;
		
		if (slotContents._queues[queueIndex]._lock.tryLock()) {
			if (!slotContents._queues[queueIndex]._queue.empty()) {
				value = slotContents._queues[queueIndex]._queue.front();
				slotContents._queues[queueIndex]._queue.pop();
				
				slotContents._queues[queueIndex]._lock.unlock();
				
				return true;
			}
		}
		
		return false;
	}
	
	
	template <typename ConsumerType>
	void consume_all(ConsumerType consumer, Slot &slot)
	{
		assert(slot != Slot());
		SlotContents &slotContents = _slots[slot];
		
#if 0
		int remaining = MAX_CONCURRENCY_PER_SLOT;
		bool done[MAX_CONCURRENCY_PER_SLOT];
		
		for (int i = 0; i < MAX_CONCURRENCY_PER_SLOT; i++) {
			done[i] = false;
		}
		
		while (remaining > 0) {
			int queueIndex = slotContents._nextPosition++;
			queueIndex = queueIndex % MAX_CONCURRENCY_PER_SLOT;
			
			if (!done[queueIndex] && slotContents._queues[queueIndex]._lock.tryLock()) {
				while (!slotContents._queues[queueIndex]._queue.empty()) {
					T &value = slotContents._queues[queueIndex]._queue.front();
					consumer(value, slot);
					slotContents._queues[queueIndex]._queue.pop();
				}
				
				remaining--;
				done[queueIndex] = true;
				
				slotContents._queues[queueIndex]._lock.unlock();
			}
		}
#else
		for (int queueIndex = 0; queueIndex < MAX_CONCURRENCY_PER_SLOT; queueIndex++) {
			slotContents._queues[queueIndex]._lock.lock();
			
			while (!slotContents._queues[queueIndex]._queue.empty()) {
				T &value = slotContents._queues[queueIndex]._queue.front();
				consumer(value, slot);
				slotContents._queues[queueIndex]._queue.pop();
			}
			
			slotContents._queues[queueIndex]._lock.unlock();
		}
#endif
	}
	
	
	template <typename ConsumerType>
	void weak_consume_all(ConsumerType consumer, Slot &slot)
	{
		assert(slot != Slot());
		SlotContents &slotContents = _slots[slot];
		
		while (true) {
			int queueIndex = slotContents._nextPosition++;
			queueIndex = queueIndex % MAX_CONCURRENCY_PER_SLOT;
			
			if (slotContents._queues[queueIndex]._lock.tryLock()) {
				while (!slotContents._queues[queueIndex]._queue.empty()) {
					T &value = slotContents._queues[queueIndex]._queue.front();
					consumer(value, slot);
					slotContents._queues[queueIndex]._queue.pop();
				}
				
				slotContents._queues[queueIndex]._lock.unlock();
				
				break;
			}
		}
	}
	
	
	template <typename ConsumerType>
	void consume_all(ConsumerType consumer)
	{
		for (int i = 0; i < _slotManager._nextFreeSlot.load(); i++) {
			Slot slot(i);
			
			consume_all(consumer, slot);
		}
	}
	
	
	template <typename ConsumerType>
	void weak_consume_all(ConsumerType consumer)
	{
		for (int i = 0; i < _slotManager._nextFreeSlot.load(); i++) {
			Slot slot(i);
			
			weak_consume_all(consumer, slot);
		}
	}
	
	
};


#endif // CONCURRENT_UNORDERED_LIST_HPP
