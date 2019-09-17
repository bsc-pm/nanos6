/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
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
	enum flag_bits {
		HAS_BEEN_DELETED_BIT=0,
		TOTAL_FLAG_BITS
	};
	typedef std::bitset<TOTAL_FLAG_BITS> flags_t;
#endif
	spinlock_t _lock;
	//! This will handle the dependencies of nested tasks.
	bottom_map_t _accessMap;
	DataAccess * _accessArray;
	void ** _addressArray;

	bool _isMain;
	size_t _num_deps;
	size_t currentIndex;

	std::atomic<int> _deletableCount;
#ifndef NDEBUG
	flags_t _flags;
#endif


	TaskDataAccesses()
		: _lock(),
		_accessMap(),
		_accessArray(nullptr),
		_addressArray(nullptr),
		_isMain(false),
		_num_deps(0),
		currentIndex(0),
		_deletableCount(0)
#ifndef NDEBUG
		, _flags()
#endif
	{
	}

	TaskDataAccesses(void *accessArray , void *addressArray, bool isMain, size_t num_deps)
		: _lock(),
		_accessMap(),
		_accessArray((DataAccess *) accessArray),
		_addressArray((void **) addressArray),
		_isMain(isMain),
		_num_deps(num_deps), currentIndex(0), _deletableCount(0)
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

		_accessMap.clear();

#ifndef NDEBUG
		hasBeenDeleted() = true;
#endif
	}

	TaskDataAccesses(TaskDataAccesses const &other) = delete;

	inline bool hasDataAccesses() const
	{
		return (_num_deps > 0);
	}

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

	inline bool decreaseDeletableCount() {
		/* We don't care about ordering, only atomicity, and that only one gets 0 as an answer */
		int res = (_deletableCount.fetch_sub(1, std::memory_order_relaxed) - 1);
		assert(res >= 0);
		return (res == 0);

	}

	inline void increaseDeletableCount() {
		_deletableCount.fetch_add(1, std::memory_order_relaxed);
	}

	inline DataAccess * findAccess(void * address) {
		for(size_t i = 0; i < currentIndex; ++i) {
			if(_addressArray[i] == address)
				return &_accessArray[i];
		}

		return nullptr;
	}

	inline size_t getRealAccessNumber() {
		return currentIndex;
	}
};

#endif // TASK_DATA_ACCESSES_HPP
