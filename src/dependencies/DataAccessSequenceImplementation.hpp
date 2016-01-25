#ifndef DATA_ACCESS_SEQUENCE_IMPLEMENTATION_HPP
#define DATA_ACCESS_SEQUENCE_IMPLEMENTATION_HPP

#include <cassert>
#include <mutex>

#include "DataAccessSequence.hpp"
#include "tasks/Task.hpp"

#include <InstrumentDependenciesByAccessSequences.hpp>
#include <InstrumentDependenciesByGroup.hpp>


DataAccessSequence::DataAccessSequence()
	: _accessRange(),
	_lock(), _accessSequence(), _superAccess(0),
	_instrumentationId(Instrument::registerAccessSequence(Instrument::data_access_id_t(), Instrument::task_id_t()))
{
}


DataAccessSequence::DataAccessSequence(DataAccessRange accessRange, DataAccess *superAccess)
	: _accessRange(accessRange),
	_lock(), _accessSequence(), _superAccess(superAccess),
	_instrumentationId(Instrument::registerAccessSequence(Instrument::data_access_id_t(), Instrument::task_id_t()))
{
}


bool DataAccessSequence::reevaluateSatisfiability(DataAccessSequence::access_sequence_t::iterator position)
{
	DataAccess &dataAccess = *position;
	
	if (dataAccess._satisfied) {
		// Already satisfied
		return false;
	}
	
	if (position == _accessSequence.begin()) {
		// The first position is satisfied, otherwise the parent task code is incorrect
		dataAccess._satisfied = true;
		return true;
	}
	
	if (dataAccess._type == WRITE_ACCESS_TYPE) {
		// A write access with accesses before it
		return false;
	}
	
	if (dataAccess._type == READWRITE_ACCESS_TYPE) {
		// A readwrite access with accesses before it
		return false;
	}
	
	--position;
	DataAccess const &previousAccess = *position;
	if (!previousAccess._satisfied) {
		// If the preceeding access is not satisfied, this cannot be either
		return false;
	}
	
	assert(dataAccess._type == READ_ACCESS_TYPE);
	assert(previousAccess._satisfied);
	if (previousAccess._type == READ_ACCESS_TYPE) {
		// Consecutive reads are satisfied together
		dataAccess._satisfied = true;
		return true;
	} else {
		assert((previousAccess._type == WRITE_ACCESS_TYPE) || (previousAccess._type == READWRITE_ACCESS_TYPE));
		// Read after Write
		return false;
	}
}


bool DataAccessSequence::upgradeAccess(Task *task, access_sequence_t::reverse_iterator &position, DataAccess &oldAccess, DataAccessType newAccessType)
{
	if (oldAccess._type == newAccessType) {
		// An identical access
		return true; // Do not count this one
	} else if ((newAccessType == WRITE_ACCESS_TYPE) && (oldAccess._type == READWRITE_ACCESS_TYPE)) {
		return true; // The old access subsumes this
	} else if ((newAccessType == READWRITE_ACCESS_TYPE) && (oldAccess._type == WRITE_ACCESS_TYPE)) {
		// An almost identical access
		Instrument::upgradedDataAccessInSequence(_instrumentationId, oldAccess._instrumentationId, oldAccess._type, newAccessType, false, task->getInstrumentationTaskId());
		oldAccess._type = newAccessType;
		
		return true; // Do not count this one
	} else if (oldAccess._type == READ_ACCESS_TYPE) {
		// Upgrade a read into a write or readwrite
		assert((newAccessType == WRITE_ACCESS_TYPE) || (newAccessType == READWRITE_ACCESS_TYPE));
		
		Instrument::removeTaskFromAccessGroup(this, task->getInstrumentationTaskId());
		Instrument::beginAccessGroup(task->getParent()->getInstrumentationTaskId(), this, false);
		Instrument::addTaskToAccessGroup(this, task->getInstrumentationTaskId());
		
		if (oldAccess._satisfied) {
			// Calculate if the upgraded access is satisfied
			--position;
			bool satisfied = (position == _accessSequence.rend());
			oldAccess._satisfied = satisfied;
			
			Instrument::upgradedDataAccessInSequence(_instrumentationId, oldAccess._instrumentationId, oldAccess._type, newAccessType, !satisfied, task->getInstrumentationTaskId());
			
			// Upgrade the access type
			oldAccess._type = READWRITE_ACCESS_TYPE;
			
			return satisfied; // A new chance for the access to not be satisfied
		} else {
			Instrument::upgradedDataAccessInSequence(_instrumentationId, oldAccess._instrumentationId, oldAccess._type, newAccessType, false, task->getInstrumentationTaskId());
			
			// Upgrade the access type
			oldAccess._type = READWRITE_ACCESS_TYPE;
			
			return true; // The predecessor has already been counted
		}
	} else {
		assert((oldAccess._type == WRITE_ACCESS_TYPE) || (oldAccess._type == READWRITE_ACCESS_TYPE));
		
		// The old access was as restrictive as possible
		return true; // Satisfiability has already been accounted for
	}
}


#endif // DATA_ACCESS_SEQUENCE_IMPLEMENTATION_HPP
