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

#define TASK_DEPS_VECTOR_CUTOFF 8

struct TaskDataAccesses {
	typedef PaddedTicketSpinLock<int, 128> spinlock_t;
	
	typedef std::unordered_map<void *, DataAccessSequence *> addresses_map_t;
	typedef std::vector<std::pair<void *, DataAccessSequence *> > addresses_vec_t;
	//typedef std::deque<void *> address_list_t;
	typedef std::vector<void *> address_list_t;
	
#ifndef NDEBUG
	enum flag_bits {
		HAS_BEEN_DELETED_BIT=0,
		TOTAL_FLAG_BITS
	};
	typedef std::bitset<TOTAL_FLAG_BITS> flags_t;
#endif

	spinlock_t _lock;
	addresses_map_t *_dataAccessSequencesMap;
	addresses_vec_t *_dataAccessSequencesVec;
	address_list_t *_accessAddresses;
    bool _map;
    bool _vecToMap;
#ifndef NDEBUG
	flags_t _flags;
#endif
	size_t _num_deps;
	
	TaskDataAccesses()
		: _lock(),
		_dataAccessSequencesMap(nullptr),
		_dataAccessSequencesVec(nullptr),
		_accessAddresses(nullptr),
        _map(false), _vecToMap(false), _num_deps(0)
#ifndef NDEBUG
		, _flags()
#endif
	{
	}

	TaskDataAccesses(void * seqs, void * addresses, bool main, size_t num_deps)
		: _lock(),
		_dataAccessSequencesMap(nullptr),
		_dataAccessSequencesVec(nullptr),
		_accessAddresses((address_list_t *)addresses),
        _map(false), _vecToMap(false), _num_deps(num_deps)
#ifndef NDEBUG
		, _flags()
#endif
	{

        if(_map)
            _dataAccessSequencesMap = (addresses_map_t *)seqs;
        else {
            _dataAccessSequencesVec = (addresses_vec_t *)seqs;
            if(seqs != nullptr)
                _dataAccessSequencesVec->clear();
        }
        // We use num_deps to prevent further allocations
        if(addresses != nullptr)
            _accessAddresses->reserve(_num_deps);
	}
	
	~TaskDataAccesses()
	{
		// We take the lock since the task may be marked for deletion while the lock is held
		std::lock_guard<spinlock_t> guard(_lock);
		assert(!hasBeenDeleted());
		
        if(_dataAccessSequencesMap != nullptr)
            _dataAccessSequencesMap->clear();
        if(_dataAccessSequencesVec != nullptr)
            _dataAccessSequencesVec->clear();
        if(_accessAddresses != nullptr)
            _accessAddresses->clear();

        destroySequences();

#ifndef NDEBUG
		hasBeenDeleted() = true;
#endif
	}
	
	TaskDataAccesses(TaskDataAccesses const &other) = delete;
	
	inline bool hasDataAccesses() const
	{
        if(_accessAddresses)
            return (!_accessAddresses->empty());
        return false;
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

    inline void destroySequences() 
    {
        if(_dataAccessSequencesMap == nullptr && _dataAccessSequencesVec == nullptr)
            return;

        if(_dataAccessSequencesMap != nullptr) 
		    for (addresses_map_t::iterator it = _dataAccessSequencesMap->begin(); it != _dataAccessSequencesMap->end(); ++it) {
		    	DataAccessSequence *sequence = it->second;
		    	assert(sequence != nullptr);
                sequence->_lock.lock();
                bool destroy = sequence->decrementRemaining() == 0;
                if(destroy) {
		    	    assert(sequence->empty());
		    	    MemoryAllocator::deleteObject<DataAccessSequence>(sequence);
                }
		    }
        else
		    for (addresses_vec_t::iterator it = _dataAccessSequencesVec->begin(); it != _dataAccessSequencesVec->end(); ++it) {
		    	DataAccessSequence *sequence = it->second;
		    	assert(sequence != nullptr);
                sequence->_lock.lock();
                bool destroy = sequence->decrementRemaining() == 0;
                if(destroy) {
		    	    assert(sequence->empty());
		    	    
		    	    MemoryAllocator::deleteObject<DataAccessSequence>(sequence);
                }
		    }

        if(_vecToMap) {
            assert(_map && _dataAccessSequencesMap != nullptr);
            MemoryAllocator::deleteObject<addresses_map_t>(_dataAccessSequencesMap);
        }

    }

    inline void vecToMap() 
    {
        assert(!_map && _dataAccessSequencesMap == nullptr && _dataAccessSequencesVec != nullptr);
        _dataAccessSequencesMap = MemoryAllocator::newObject<addresses_map_t>(_dataAccessSequencesVec->begin(), _dataAccessSequencesVec->end()); 
        _dataAccessSequencesVec = nullptr;
        _map = true;
        assert(_map && _dataAccessSequencesMap != nullptr);
    }
};

#endif // TASK_DATA_ACCESSES_HPP
