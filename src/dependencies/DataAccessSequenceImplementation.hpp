#ifndef DATA_ACCESS_SEQUENCE_IMPLEMENTATION_HPP
#define DATA_ACCESS_SEQUENCE_IMPLEMENTATION_HPP

#include <cassert>
#include <mutex>

#include "DataAccessSequence.hpp"
#include "tasks/Task.hpp"

#include <InstrumentDependenciesByAccessLinks.hpp>


inline DataAccessSequence::DataAccessSequence(SpinLock *lock)
	: _accessRange(),
	_superAccess(0), _lock(lock), _accessSequence()
{
}


inline DataAccessSequence::DataAccessSequence(DataAccessRange accessRange, SpinLock *lock)
	: _accessRange(accessRange),
	_superAccess(nullptr), _lock(lock), _accessSequence()
{
}


inline DataAccessSequence::DataAccessSequence(DataAccessRange accessRange, DataAccess *superAccess, SpinLock *lock)
	: _accessRange(accessRange),
	_superAccess(superAccess), _lock(lock), _accessSequence()
{
}


inline std::unique_lock<SpinLock> DataAccessSequence::getLockGuard()
{
	assert(_lock != nullptr);
	return std::unique_lock<SpinLock>(*_lock);
}


DataAccess *DataAccessSequence::getEffectivePrevious(DataAccess *dataAccess)
{
	DataAccessSequence *currentSequence = this;
	DataAccessSequence::access_sequence_t::iterator next;
	
	if (dataAccess != nullptr) {
		next = _accessSequence.iterator_to(*dataAccess);
	} else {
		// Looking for the effective previous to a new access that has yet to be added and assuming that the sequence is empty
		assert(_accessSequence.empty());
		next = _accessSequence.begin();
	}
	
	// While we hit the top of a sequence, go to the previous of the parent
	while (next == currentSequence->_accessSequence.begin()) {
		DataAccess *superAccess = currentSequence->_superAccess;
		
		if (superAccess == nullptr) {
			// We reached the beginning of the logical sequence or the top of the root sequence
			return nullptr;
		}
		
		currentSequence = superAccess->_dataAccessSequence;
		next = currentSequence->_accessSequence.iterator_to(*superAccess);
	}
	
	assert(next != currentSequence->_accessSequence.begin());
	
	next--;
	DataAccess *effectivePrevious = &(*next);
	
	return effectivePrevious;
}


#include "DataAccessImplementation.hpp"


#endif // DATA_ACCESS_SEQUENCE_IMPLEMENTATION_HPP
