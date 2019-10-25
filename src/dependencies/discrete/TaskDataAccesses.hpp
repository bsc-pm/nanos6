/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_DATA_ACCESSES_HPP
#define TASK_DATA_ACCESSES_HPP

#include <atomic>
#include <cassert>
#include <mutex>
#include <unordered_map>
#include <array>

#include "lowlevel/TicketSpinLock.hpp"
#include "BottomMapEntry.hpp"
#include <MemoryAllocator.hpp>

struct DataAccess;

struct TaskDataAccesses {
	typedef TicketSpinLock<int> spinlock_t;
	typedef std::unordered_map<void *, BottomMapEntry> bottom_map_t;
	
#ifndef NDEBUG
	enum flag_bits_t {
		HAS_BEEN_DELETED_BIT=0,
		TOTAL_FLAG_BITS
	};
	typedef std::bitset<TOTAL_FLAG_BITS> flags_t;
#endif
	spinlock_t _lock;
	//! This will handle the dependencies of nested tasks.
	bottom_map_t _subaccessBottomMap;
	DataAccess * _accessArray;
	void ** _addressArray;
	size_t _maxDeps;
	size_t _currentIndex;
	
	std::atomic<int> _deletableCount;
#ifndef NDEBUG
	flags_t _flags;
#endif


	TaskDataAccesses()
		: _lock(),
		_subaccessBottomMap(),
		_accessArray(nullptr),
		_addressArray(nullptr),
		_maxDeps(0),
		_currentIndex(0),
		_deletableCount(0)
#ifndef NDEBUG
		, _flags()
#endif
	{
	}
	
	TaskDataAccesses(void *accessArray , void *addressArray, size_t maxDeps)
		: _lock(),
		_subaccessBottomMap(),
		_accessArray((DataAccess *) accessArray),
		_addressArray((void **) addressArray),
		_maxDeps(maxDeps), _currentIndex(0), _deletableCount(0)
#ifndef NDEBUG
		, _flags()
#endif
	{
	}
	
	~TaskDataAccesses()
	{
		// We take the lock since the task may be marked for deletion while the lock is held
		std::lock_guard<spinlock_t> guard(_lock);
		assert(!hasBeenDeleted());
		
		_subaccessBottomMap.clear();
		
#ifndef NDEBUG
		hasBeenDeleted() = true;
#endif
	}
	
	TaskDataAccesses(TaskDataAccesses const &other) = delete;
	
#ifndef NDEBUG
	bool hasBeenDeleted() const
	{
		return _flags[HAS_BEEN_DELETED_BIT];
	}
	
	flags_t::reference hasBeenDeleted()
	{
		return _flags[HAS_BEEN_DELETED_BIT];
	}
#endif

	inline bool decreaseDeletableCount()
	{
		/* We don't care about ordering, only atomicity, and that only one gets 0 as an answer */
		int res = (_deletableCount.fetch_sub(1, std::memory_order_relaxed) - 1);
		assert(res >= 0);
		return (res == 0);
		
	}
	
	inline void increaseDeletableCount()
	{
		_deletableCount.fetch_add(1, std::memory_order_relaxed);
	}
	
	inline DataAccess * findAccess(void * address) const
	{
		for(size_t i = 0; i < _currentIndex; ++i) {
			if(_addressArray[i] == address)
				return &_accessArray[i];
		}
		
		return nullptr;
	}
	
	inline size_t getRealAccessNumber() const
	{
		return _currentIndex;
	}
	
	inline bool hasDataAccesses() const
	{
		return (getRealAccessNumber() > 0);
	}
	
	inline size_t getAdditionalMemorySize() const
	{
		return (sizeof(DataAccess) + sizeof(void *)) * _maxDeps;
	}
};

#endif // TASK_DATA_ACCESSES_HPP
