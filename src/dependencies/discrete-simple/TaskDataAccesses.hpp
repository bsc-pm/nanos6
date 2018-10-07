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

#include "DataAccessSequence.hpp"
#include "lowlevel/PaddedTicketSpinLock.hpp"
#include <MemoryAllocator.hpp>


struct DataAccess;


struct TaskDataAccesses {
	typedef PaddedTicketSpinLock<int, 128> spinlock_t;
	
	typedef std::unordered_map<void *, DataAccessSequence *> addresses_t;
	typedef std::deque<void *> address_list_t;
	
#ifndef NDEBUG
	enum flag_bits {
		HAS_BEEN_DELETED_BIT=0,
		TOTAL_FLAG_BITS
	};
	typedef std::bitset<TOTAL_FLAG_BITS> flags_t;
#endif
		
	spinlock_t _lock;
	addresses_t _dataAccessSequences;
	std::deque<void *> _readAccessAddresses;
	std::deque<void *> _writeAccessAddresses;
#ifndef NDEBUG
	flags_t _flags;
#endif
	
	TaskDataAccesses()
		: _lock(),
		_dataAccessSequences(),
		_readAccessAddresses(),
		_writeAccessAddresses()
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
		
		addresses_t::iterator it;
		for (it = _dataAccessSequences.begin(); it != _dataAccessSequences.end(); ++it) {
			DataAccessSequence *sequence = it->second;
			assert(sequence != nullptr);
			assert(sequence->empty());
			
			MemoryAllocator::deleteObject<DataAccessSequence>(sequence);
		}
		_dataAccessSequences.clear();
		_readAccessAddresses.clear();
		_writeAccessAddresses.clear();

#ifndef NDEBUG
		hasBeenDeleted() = true;
#endif
	}
	
	TaskDataAccesses(TaskDataAccesses const &other) = delete;
	
	inline bool hasDataAccesses() const
	{
		return (!_readAccessAddresses.empty() || !_writeAccessAddresses.empty());
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
};

#endif // TASK_DATA_ACCESSES_HPP
