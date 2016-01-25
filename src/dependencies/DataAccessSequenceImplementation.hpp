#ifndef DATA_ACCESS_SEQUENCE_IMPLEMENTATION_HPP
#define DATA_ACCESS_SEQUENCE_IMPLEMENTATION_HPP

#include <cassert>
#include <mutex>

#include "DataAccessSequence.hpp"
#include "tasks/Task.hpp"

#include <InstrumentDependenciesByAccessSequences.hpp>
#include <InstrumentDependenciesByGroup.hpp>


inline DataAccessSequence::DataAccessSequence()
	: _accessRange(),
	_superAccess(0), _rootLock(), _accessSequence(),
	_instrumentationId(Instrument::registerAccessSequence(Instrument::data_access_id_t(), Instrument::task_id_t()))
{
}


inline DataAccessSequence::DataAccessSequence(DataAccessRange accessRange)
	: _accessRange(accessRange),
	_superAccess(nullptr), _rootLock(), _accessSequence(),
	_instrumentationId(Instrument::registerAccessSequence(Instrument::data_access_id_t(), Instrument::task_id_t()))
{
}


inline DataAccessSequence::DataAccessSequence(DataAccessRange accessRange, DataAccess *superAccess)
	: _accessRange(accessRange),
	_superAccess(superAccess), _rootSequence(superAccess->_dataAccessSequence->getRootSequence()), _accessSequence(),
	_instrumentationId(Instrument::registerAccessSequence(Instrument::data_access_id_t(), Instrument::task_id_t()))
{
}


inline DataAccessSequence::DataAccessSequence(DataAccessRange accessRange, DataAccess *superAccess, DataAccessSequence *rootSequence)
	: _accessRange(accessRange),
	_superAccess(superAccess), _rootSequence(rootSequence), _accessSequence(),
	_instrumentationId(Instrument::registerAccessSequence(Instrument::data_access_id_t(), Instrument::task_id_t()))
{
}


inline DataAccessSequence::~DataAccessSequence()
{
	if (_superAccess == nullptr) {
		_rootLock.~SpinLock();
	}
	_accessSequence.~list_impl();
}


inline DataAccessSequence *DataAccessSequence::getRootSequence()
{
	if (_superAccess == nullptr) {
		return this;
	} else {
		assert(_rootSequence != nullptr);
		assert(_rootSequence->_superAccess == nullptr);
		return _rootSequence;
	}
}


inline void DataAccessSequence::lock()
{
	getRootSequence()->_rootLock.lock();
}

inline void DataAccessSequence::unlock()
{
	getRootSequence()->_rootLock.unlock();
}


inline std::unique_lock<SpinLock> DataAccessSequence::getLockGuard()
{
	return std::unique_lock<SpinLock>(getRootSequence()->_rootLock);
}


inline bool DataAccessSequence::evaluateSatisfiability(DataAccess *previousDataAccess, DataAccessType nextAccessType)
{
	if (previousDataAccess == nullptr) {
		// The first position is satisfied
		return true;
	}
	
	if (!previousDataAccess->_satisfied) {
		// If the preceeding access is not satisfied, this cannot be either
		return false;
	}
	
	if (nextAccessType == WRITE_ACCESS_TYPE) {
		// A write access with accesses before it
		return false;
	}
	
	if (nextAccessType == READWRITE_ACCESS_TYPE) {
		// A readwrite access with accesses before it
		return false;
	}
	
	assert(nextAccessType == READ_ACCESS_TYPE);
	assert(previousDataAccess->_satisfied);
	if (previousDataAccess->_type == READ_ACCESS_TYPE) {
		// Consecutive reads are satisfied together
		return true;
	} else {
		assert((previousDataAccess->_type == WRITE_ACCESS_TYPE) || (previousDataAccess->_type == READWRITE_ACCESS_TYPE));
		// Read after Write
		return false;
	}
}


inline bool DataAccessSequence::reevaluateSatisfiability(DataAccess *previousDataAccess, DataAccess *targetDataAccess)
{
	if (targetDataAccess->_satisfied) {
		// Already satisfied
		return false;
	}
	
	return DataAccessSequence::evaluateSatisfiability(previousDataAccess, targetDataAccess->_type);
}


bool DataAccessSequence::upgradeAccess(Task *task, DataAccess *dataAccess, DataAccessType newAccessType)
{
	assert(dataAccess != nullptr);
	
	if (dataAccess->_type == newAccessType) {
		// An identical access
		return true; // Do not count this one
	} else if ((newAccessType == WRITE_ACCESS_TYPE) && (dataAccess->_type == READWRITE_ACCESS_TYPE)) {
		return true; // The old access subsumes this
	} else if ((newAccessType == READWRITE_ACCESS_TYPE) && (dataAccess->_type == WRITE_ACCESS_TYPE)) {
		// An almost identical access
		Instrument::upgradedDataAccessInSequence(_instrumentationId, dataAccess->_instrumentationId, dataAccess->_type, newAccessType, false, task->getInstrumentationTaskId());
		dataAccess->_type = newAccessType;
		
		return true; // Do not count this one
	} else if (dataAccess->_type == READ_ACCESS_TYPE) {
		// Upgrade a read into a write or readwrite
		assert((newAccessType == WRITE_ACCESS_TYPE) || (newAccessType == READWRITE_ACCESS_TYPE));
		
		Instrument::removeTaskFromAccessGroup(this, task->getInstrumentationTaskId());
		Instrument::beginAccessGroup(task->getParent()->getInstrumentationTaskId(), this, false);
		Instrument::addTaskToAccessGroup(this, task->getInstrumentationTaskId());
		
		DataAccessType oldAccessType = dataAccess->_type;
		
		// Upgrade the access type
		dataAccess->_type = READWRITE_ACCESS_TYPE;
		
		if (dataAccess->_satisfied) {
			// Calculate if the satisfiability of the upgraded access
			DataAccess *effectivePrevious = getEffectivePrevious(dataAccess);
			bool satisfied = evaluateSatisfiability(effectivePrevious, dataAccess->_type);
			dataAccess->_satisfied = satisfied;
			
			Instrument::upgradedDataAccessInSequence(_instrumentationId, dataAccess->_instrumentationId, oldAccessType, newAccessType, !satisfied, task->getInstrumentationTaskId());
			
			return satisfied; // A new chance for the access to not be satisfied
		} else {
			Instrument::upgradedDataAccessInSequence(_instrumentationId, dataAccess->_instrumentationId, dataAccess->_type, newAccessType, false, task->getInstrumentationTaskId());
			
			return true; // The predecessor has already been counted
		}
	} else {
		assert((dataAccess->_type == WRITE_ACCESS_TYPE) || (dataAccess->_type == READWRITE_ACCESS_TYPE));
		
		// The old access was as restrictive as possible
		return true; // Satisfiability has already been accounted for
	}
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


#endif // DATA_ACCESS_SEQUENCE_IMPLEMENTATION_HPP
