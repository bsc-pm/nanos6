/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_DATA_ACCESSES_HPP
#define TASK_DATA_ACCESSES_HPP

#include <atomic>
#include <cassert>
#include <functional>
#include <mutex>

#include "BottomMapEntry.hpp"
#include "CommutativeSemaphore.hpp"
#include "TaskDataAccessesInfo.hpp"
#include "lowlevel/TicketSpinLock.hpp"
#include "support/Containers.hpp"

#include <DependencySystem.hpp>
#include <MemoryAllocator.hpp>

struct DataAccess;

struct TaskDataAccesses {
	typedef Container::unordered_map<void *, BottomMapEntry> bottom_map_t;
	typedef Container::unordered_map<void *, DataAccess> access_map_t;

#ifndef NDEBUG
	enum flag_bits_t {
		HAS_BEEN_DELETED_BIT = 0,
		TOTAL_FLAG_BITS
	};
	typedef std::bitset<TOTAL_FLAG_BITS> flags_t;
#endif
	//! This will handle the dependencies of nested tasks.
	bottom_map_t _subaccessBottomMap;
	DataAccess *_accessArray;
	void **_addressArray;
	size_t _maxDeps;
	size_t _currentIndex;
	CommutativeSemaphore::commutative_mask_t _commutativeMask;

	std::atomic<int> _deletableCount;
	access_map_t *_accessMap;
#ifndef NDEBUG
	flags_t _flags;
#endif


	TaskDataAccesses() :
		_subaccessBottomMap(),
		_accessArray(nullptr),
		_addressArray(nullptr),
		_maxDeps(0),
		_currentIndex(0),
		_commutativeMask(0),
		_deletableCount(0),
		_accessMap(nullptr)
#ifndef NDEBUG
		,
		_flags()
#endif
	{
	}

	TaskDataAccesses(TaskDataAccessesInfo taskAccessInfo) :
		_subaccessBottomMap(),
		_accessArray(taskAccessInfo.getAccessArrayLocation()),
		_addressArray(taskAccessInfo.getAddressArrayLocation()),
		_maxDeps(taskAccessInfo.getNumDeps()),
		_currentIndex(0),
		_deletableCount(0),
		_accessMap(nullptr)
#ifndef NDEBUG
		,
		_flags()
#endif
	{
		if (_maxDeps > ACCESS_LINEAR_CUTOFF) {
			_accessMap = MemoryAllocator::newObject<access_map_t>();
			assert(_accessMap != nullptr);

			_accessMap->reserve((_maxDeps != (size_t) -1) ? _maxDeps : ACCESS_LINEAR_CUTOFF);
		}
	}

	~TaskDataAccesses()
	{
		assert(!hasBeenDeleted());

		if (_accessMap != nullptr) {
			MemoryAllocator::deleteObject(_accessMap);
		}

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
		int res = (_deletableCount.fetch_sub(1, std::memory_order_relaxed) - 1);
		assert(res >= 0);
		return (res == 0);
	}

	inline void increaseDeletableCount()
	{
		__attribute__((unused)) int res = _deletableCount.fetch_add(1, std::memory_order_relaxed);
		assert(res >= 0);
	}

	inline DataAccess *findAccess(void *address) const
	{
		if (_accessMap != nullptr) {
			access_map_t::iterator itAccess = _accessMap->find(address);
			if (itAccess != _accessMap->end())
				return &itAccess->second;
		} else {
			for (size_t i = 0; i < _currentIndex; ++i) {
				if (_addressArray[i] == address)
					return &_accessArray[i];
			}
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
		TaskDataAccessesInfo info(_maxDeps);
		return info.getAllocationSize();
	}

	inline DataAccess *allocateAccess(void *address, DataAccessType type, Task *originator, size_t length, bool weak, bool &existing)
	{
		if (_accessMap != nullptr) {
			std::pair<access_map_t::iterator, bool> emplaced = _accessMap->emplace(std::piecewise_construct,
				std::forward_as_tuple(address),
				std::forward_as_tuple(type, originator, address, length, weak));

			existing = !emplaced.second;
			if (!existing)
				_currentIndex++;
			return &emplaced.first->second;
		} else {
			DataAccess *ret = findAccess(address);
			existing = (ret != nullptr);
			assert(_currentIndex < _maxDeps);

			if (!existing) {
				_addressArray[_currentIndex] = address;
				ret = &_accessArray[_currentIndex++];
				new (ret) DataAccess(type, originator, address, length, weak);
			}

			return ret;
		}
	}

	inline bool forAll(std::function<bool(void *, DataAccess *)> callback)
	{
		if (_accessMap != nullptr) {
			access_map_t::iterator itAccess = _accessMap->begin();

			while (itAccess != _accessMap->end()) {
				if (!callback(itAccess->first, &itAccess->second))
					return false;
				itAccess++;
			}
		} else {
			for (size_t i = 0; i < getRealAccessNumber(); ++i) {
				if (!callback(_addressArray[i], &_accessArray[i]))
					return false;
			}
		}

		return true;
	}
};

#endif // TASK_DATA_ACCESSES_HPP
