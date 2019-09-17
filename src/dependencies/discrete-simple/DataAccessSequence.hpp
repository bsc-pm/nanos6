/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_SEQUENCE_HPP
#define DATA_ACCESS_SEQUENCE_HPP

#include <atomic>
#include <bitset>
#include <cassert>
#include <deque>
#include <vector>
#include <set>

#include "CPUDependencyData.hpp"
#include "DataAccess.hpp"
#include "lowlevel/PaddedTicketSpinLock.hpp"

class Task;


struct DataAccessSequence {
public:
	typedef PaddedTicketSpinLock<int, 128> spinlock_t;
	
	spinlock_t _lock;
	
private:
	//typedef std::deque<DataAccess> sequence_t;
	typedef std::vector<DataAccess> sequence_t;
	typedef CPUDependencyData::satisfied_originator_list_t satisfied_originator_list_t;
	
	//! The sequence of accesses
	sequence_t _sequence;
	
	//! Number of satisfied reader tasks at the sequence
	unsigned int _satisfiedReaders;
	
	//! Number of uncompleted satisfied reader tasks at the sequence
	unsigned int _uncompletedReaders;
	
	//! Number of uncompleted satisfied reader tasks at the sequence
	unsigned int _uncompletedWriters;

    //! The number of tasks that are pointing this sequence.
    size_t _remaining;
	
public:
	DataAccessSequence()
		: _lock(),
		_sequence(),
		_satisfiedReaders(0),
		_uncompletedReaders(0),
		_uncompletedWriters(0),
        _remaining(0)
	{
        _sequence.reserve(8);
	}
	
	DataAccessSequence(const DataAccessSequence &other) = delete;
	
	~DataAccessSequence()
	{
		assert(_satisfiedReaders == 0);
		assert(_uncompletedReaders == 0);
		assert(_uncompletedWriters == 0);
		assert(_sequence.empty());
	}
	
	inline bool empty()
	{
		return _sequence.empty();
	}
	
	inline bool registeredLastDataAccess(Task *task)
	{
		if (!_sequence.empty()) {
			return (_sequence.back().getOriginator() == task);
		}
		return false;
	}
	
	inline bool upgradeLastDataAccess(DataAccessType *previousType)
	{
		assert(previousType != nullptr);
		assert(!_sequence.empty());
		
		DataAccess &dataAccess = _sequence.back();
		
		DataAccessType type = dataAccess.getType();
		*previousType = type;
		
		bool wasSatisfied = isLastDataAccessSatisfied();
		dataAccess.setType(READWRITE_ACCESS_TYPE);
		
		if (wasSatisfied && type == READ_ACCESS_TYPE) {
			--_satisfiedReaders;
			--_uncompletedReaders;
			
			dataAccess.setType(READWRITE_ACCESS_TYPE);
			
			if (_satisfiedReaders == 0) {
				++_uncompletedWriters;
				return false;
			}
			return true;
		}
		return false;
	}
	
	inline bool registerDataAccess(DataAccessType accessType, Task *task)
	{
		unsigned int prevSize = _sequence.size();
		_sequence.emplace_back(accessType, task);
		
		bool satisfied = true;
		if (accessType == READ_ACCESS_TYPE && prevSize == _satisfiedReaders) {
			_satisfiedReaders += 1;
			_uncompletedReaders += 1;
		} else if (accessType != READ_ACCESS_TYPE && prevSize == 0) {
			_uncompletedWriters += 1;
		} else {
			satisfied = false;
		}
		
		return satisfied;
	}
	
	inline void finalizeDataAccess(__attribute__((unused)) Task *task, DataAccessType accessType, satisfied_originator_list_t &satisfiedOriginators)
	{
		assert(!_sequence.empty());
		
		unsigned int remainingSatisfied = 0;
		if (accessType == READ_ACCESS_TYPE) {
			assert(_uncompletedWriters == 0);
			remainingSatisfied = --_uncompletedReaders;
		} else {
			assert(_satisfiedReaders == 0);
			assert(_uncompletedReaders == 0);
			remainingSatisfied = --_uncompletedWriters;
		}
		
		if (remainingSatisfied == 0) {
			removeCompleteSuccessors(accessType);
			satisfyOldestSuccessors(satisfiedOriginators);
		}
	}

    inline void incrementRemaining() 
    {
        _remaining++;
    }

    inline size_t decrementRemaining()
    {
        return --_remaining;
    }
	
private:
	inline bool isLastDataAccessSatisfied()
	{
		return (_sequence.size() == 1) || (_sequence.size() == _satisfiedReaders);
	}
	
	void removeCompleteSuccessors(DataAccessType type);
	
	void satisfyOldestSuccessors(satisfied_originator_list_t &satisfiedOriginators);
};


#endif // DATA_ACCESS_HPP
