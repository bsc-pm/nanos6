#ifndef DATA_ACCESS_SEQUENCE_IMPLEMENTATION_HPP
#define DATA_ACCESS_SEQUENCE_IMPLEMENTATION_HPP

#include <cassert>
#include <mutex>

#include "DataAccessSequence.hpp"
#include "tasks/Task.hpp"

#include <InstrumentDependenciesByAccessSequences.hpp>
#include <InstrumentDependenciesByGroup.hpp>


inline DataAccessSequence::DataAccessSequence(SpinLock *lock)
	: _accessRange(),
	_superAccess(0), _lock(lock), _accessSequence(),
	_instrumentationId(Instrument::registerAccessSequence(Instrument::data_access_id_t(), Instrument::task_id_t()))
{
}


inline DataAccessSequence::DataAccessSequence(DataAccessRange accessRange, SpinLock *lock)
	: _accessRange(accessRange),
	_superAccess(nullptr), _lock(lock), _accessSequence(),
	_instrumentationId(Instrument::registerAccessSequence(Instrument::data_access_id_t(), Instrument::task_id_t()))
{
}


inline DataAccessSequence::DataAccessSequence(DataAccessRange accessRange, DataAccess *superAccess, SpinLock *lock)
	: _accessRange(accessRange),
	_superAccess(superAccess), _lock(lock), _accessSequence(),
	_instrumentationId(Instrument::registerAccessSequence(Instrument::data_access_id_t(), Instrument::task_id_t()))
{
}


inline std::unique_lock<SpinLock> DataAccessSequence::getLockGuard()
{
	assert(_lock != nullptr);
	return std::unique_lock<SpinLock>(*_lock);
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


bool DataAccessSequence::upgradeSameTypeAccess(Task *task, DataAccess /* INOUT */ *dataAccess, bool newAccessWeakness)
{
	if (dataAccess->_weak != newAccessWeakness) {
		Instrument::upgradedDataAccessInSequence(
			_instrumentationId, dataAccess->_instrumentationId,
			dataAccess->_type, dataAccess->_weak,
			dataAccess->_type, (dataAccess->_weak && newAccessWeakness),
			false, task->getInstrumentationTaskId()
		);
		dataAccess->_weak &= newAccessWeakness; // In fact, just false
	}
	
	// An identical access
	return true; // Do not count this one
}


bool DataAccessSequence::upgradeSameStrengthAccess(Task *task, DataAccess /* INOUT */ *dataAccess, DataAccessType newAccessType)
{
	if (dataAccess->_type == READWRITE_ACCESS_TYPE) {
		// The old access is as restrictive as possible
		return true;
	} else if (dataAccess->_type == WRITE_ACCESS_TYPE) {
		// A write that becomes readwrite
		assert((newAccessType == READWRITE_ACCESS_TYPE) || (newAccessType == READ_ACCESS_TYPE));
		Instrument::upgradedDataAccessInSequence(
			_instrumentationId, dataAccess->_instrumentationId,
			dataAccess->_type, dataAccess->_weak,
			READWRITE_ACCESS_TYPE, dataAccess->_weak,
			false, task->getInstrumentationTaskId()
		);
		dataAccess->_type = newAccessType;
		
		// The essential type of access did not change, and thus neither did its satisfiability
		return true; // Do not count this one
	} else {
		// Upgrade a read into a readwrite
		assert(dataAccess->_type == READ_ACCESS_TYPE);
		assert((newAccessType == WRITE_ACCESS_TYPE) || (newAccessType == READWRITE_ACCESS_TYPE));
		
		if (!dataAccess->_weak) {
			Instrument::removeTaskFromAccessGroup(this, task->getInstrumentationTaskId());
			Instrument::beginAccessGroup(task->getParent()->getInstrumentationTaskId(), this, false);
			Instrument::addTaskToAccessGroup(this, task->getInstrumentationTaskId());
		}
		
		DataAccessType oldAccessType = dataAccess->_type;
		
		// Upgrade the access type
		dataAccess->_type = READWRITE_ACCESS_TYPE;
		
		if (dataAccess->_satisfied) {
			// Calculate if the satisfiability of the upgraded access
			DataAccess *effectivePrevious = getEffectivePrevious(dataAccess);
			bool satisfied = evaluateSatisfiability(effectivePrevious, dataAccess->_type);
			dataAccess->_satisfied = satisfied;
			
			Instrument::upgradedDataAccessInSequence(
				_instrumentationId, dataAccess->_instrumentationId,
				oldAccessType, dataAccess->_weak,
				newAccessType, dataAccess->_weak,
				!satisfied, task->getInstrumentationTaskId()
			);
			
			return satisfied; // A new chance for the access to not be satisfied
		} else {
			Instrument::upgradedDataAccessInSequence(
				_instrumentationId, dataAccess->_instrumentationId,
				dataAccess->_type, dataAccess->_weak,
				newAccessType, dataAccess->_weak,
				false, task->getInstrumentationTaskId()
			);
			
			return true; // The old access has already been counted
		}
	}
}


bool DataAccessSequence::upgradeStrongAccessWithWeak(Task *task, DataAccess /* INOUT */ * /* INOUT */ &dataAccess, DataAccessType newAccessType)
{
	if (dataAccess->_type == READWRITE_ACCESS_TYPE) {
		// The old access is as restrictive as possible
		return true;
	} else if (dataAccess->_type == WRITE_ACCESS_TYPE) {
		// A write that becomes readwrite
		assert((newAccessType == READWRITE_ACCESS_TYPE) || (newAccessType == READ_ACCESS_TYPE));
		Instrument::upgradedDataAccessInSequence(
			_instrumentationId, dataAccess->_instrumentationId,
			dataAccess->_type, false,
			READWRITE_ACCESS_TYPE, false,
			false, task->getInstrumentationTaskId()
		);
		dataAccess->_type = newAccessType;
		
		// The essential type of access did not change, and thus neither did its satisfiability
		return true; // Do not count this one
	} else {
		assert(dataAccess->_type == READ_ACCESS_TYPE);
		
		if (newAccessType == READ_ACCESS_TYPE) {
			return true;
		} else {
			bool satisfied = evaluateSatisfiability(dataAccess, newAccessType);
			
			DataAccessSequence *accessSequence = dataAccess->_dataAccessSequence;
			assert(accessSequence != nullptr);
			
			Instrument::data_access_id_t newDataAccessInstrumentationId =
			Instrument::addedDataAccessInSequence(accessSequence->_instrumentationId, newAccessType, true, satisfied, task->getInstrumentationTaskId());
			
			dataAccess = new DataAccess(accessSequence, newAccessType, true, satisfied, task, accessSequence->_accessRange, newDataAccessInstrumentationId);
			accessSequence->_accessSequence.push_back(*dataAccess); // NOTE: It actually does get the pointer
			
			return satisfied;
		}
	}
}


bool DataAccessSequence::upgradeWeakAccessWithStrong(Task *task, DataAccess /* INOUT */ * /* INOUT */ &dataAccess, DataAccessType newAccessType)
{
	if (newAccessType != READ_ACCESS_TYPE) {
		// A new write or readwrite that subsumes a weak access
		assert((dataAccess->_type != WRITE_ACCESS_TYPE) || (newAccessType != WRITE_ACCESS_TYPE)); // Handled elsewhere
		
		DataAccessType oldAccessType = dataAccess->_type;
		
		// Upgrade the access type
		dataAccess->_type = READWRITE_ACCESS_TYPE;
		dataAccess->_weak = false;
		
		if (dataAccess->_satisfied) {
			// Calculate if the satisfiability of the upgraded access
			DataAccess *effectivePrevious = getEffectivePrevious(dataAccess);
			bool satisfied = evaluateSatisfiability(effectivePrevious, dataAccess->_type);
			dataAccess->_satisfied = satisfied;
			
			Instrument::upgradedDataAccessInSequence(
				_instrumentationId, dataAccess->_instrumentationId,
				oldAccessType, true,
				newAccessType, false,
				!satisfied, task->getInstrumentationTaskId()
			);
			
			return satisfied; // A new chance for the access to not be satisfied
		} else {
			Instrument::upgradedDataAccessInSequence(
				_instrumentationId, dataAccess->_instrumentationId,
				dataAccess->_type, true,
				newAccessType, false,
				false, task->getInstrumentationTaskId()
			);
			
			return true;
		}
	} else {
		// A new "strong" read to be combined with an already existing weak access
		assert(newAccessType == READ_ACCESS_TYPE);
		
		if (dataAccess->_type == READ_ACCESS_TYPE) {
			dataAccess->_weak = false;
			
			Instrument::upgradedDataAccessInSequence(
				_instrumentationId, dataAccess->_instrumentationId,
				READ_ACCESS_TYPE, true,
				READ_ACCESS_TYPE, false,
				false, task->getInstrumentationTaskId()
			);
			
			return dataAccess->_satisfied; // A new chance for the access to be accounted
		} else {
			// The new "strong" read must come before the old weak write or weak readwrite
			
			DataAccess *effectivePrevious = getEffectivePrevious(dataAccess);
			bool satisfied = evaluateSatisfiability(effectivePrevious, newAccessType);
			
			DataAccessSequence *accessSequence = dataAccess->_dataAccessSequence;
			assert(accessSequence != nullptr);
			
			// We overwrite the old DataAccess object with the "strong" read and create a new DataAccess after it with the old weak access information
			// This simplifies insertion and the instrumentation
			
			// Instrumentation for the upgrade of the existing access to "strong" read
			Instrument::upgradedDataAccessInSequence(
				_instrumentationId, dataAccess->_instrumentationId,
				dataAccess->_type, true,
				READ_ACCESS_TYPE, false,
				false, task->getInstrumentationTaskId()
			);
			if (dataAccess->_satisfied != satisfied) {
				Instrument::dataAccessBecomesSatisfied(
					accessSequence->_instrumentationId, dataAccess->_instrumentationId,
					task->getInstrumentationTaskId(), task->getInstrumentationTaskId()
				);
			}
			
			// Update existing access to "strong" read
			DataAccessType oldAccessType = dataAccess->_type;
			dataAccess->_type = READ_ACCESS_TYPE;
			dataAccess->_weak = false;
			dataAccess->_satisfied = satisfied;
			
			// New object with the old information
			Instrument::data_access_id_t newDataAccessInstrumentationId =
			Instrument::addedDataAccessInSequence(accessSequence->_instrumentationId, oldAccessType, true, false, task->getInstrumentationTaskId());
			
			dataAccess = new DataAccess(accessSequence, oldAccessType, true, false, task, accessSequence->_accessRange, newDataAccessInstrumentationId);
			accessSequence->_accessSequence.push_back(*dataAccess); // NOTE: It actually does get the pointer
			
			return satisfied;
		}
	}
}


bool DataAccessSequence::upgradeAccess(Task *task, DataAccess /* INOUT */ * /* INOUT */ &dataAccess, DataAccessType newAccessType, bool newAccessWeakness)
{
	assert(dataAccess != nullptr);
	
	if (dataAccess->_type == newAccessType) {
		return upgradeSameTypeAccess(task, dataAccess, newAccessWeakness);
	} else if (dataAccess->_weak == newAccessWeakness) {
		// Either both weak or both "strong"
		return upgradeSameStrengthAccess(task, dataAccess, newAccessType);
	} else if (!dataAccess->_weak) {
		// Current is "strong", new is weak
		assert(newAccessWeakness);
		return upgradeStrongAccessWithWeak(task, dataAccess, newAccessType);
	} else {
		// Current is weak, new is "strong"
		assert(dataAccess->_weak);
		assert(!newAccessWeakness);
		
		return upgradeWeakAccessWithStrong(task, dataAccess, newAccessType);
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
