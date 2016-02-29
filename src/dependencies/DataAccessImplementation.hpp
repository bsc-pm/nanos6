#ifndef DATA_ACCESS_IMPLEMENTATION_HPP
#define DATA_ACCESS_IMPLEMENTATION_HPP

#include <cassert>
#include <mutex>

#include "DataAccess.hpp"
#include "DataAccessSequence.hpp"
#include "tasks/Task.hpp"

#include <InstrumentDependenciesByAccessLinks.hpp>
#include <InstrumentDependenciesByGroup.hpp>


inline bool DataAccess::evaluateSatisfiability(DataAccess *effectivePrevious, DataAccessType nextAccessType)
{
	if (effectivePrevious == nullptr) {
		// The first position is satisfied
		return true;
	}
	
	if (!effectivePrevious->_satisfied) {
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
	assert(effectivePrevious->_satisfied);
	if (effectivePrevious->_type == READ_ACCESS_TYPE) {
		// Consecutive reads are satisfied together
		return true;
	} else {
		assert((effectivePrevious->_type == WRITE_ACCESS_TYPE) || (effectivePrevious->_type == READWRITE_ACCESS_TYPE));
		// Read after Write
		return false;
	}
}


inline bool DataAccess::reevaluateSatisfiability(DataAccess *effectivePrevious)
{
	if (_satisfied) {
		// Already satisfied
		return false;
	}
	
	return DataAccess::evaluateSatisfiability(effectivePrevious, _type);
}


bool DataAccess::upgradeSameTypeAccess(Task *task, DataAccess /* INOUT */ *dataAccess, bool newAccessWeakness)
{
	assert(dataAccess != nullptr);
	
	DataAccessSequence *accessSequence = dataAccess->_dataAccessSequence;
	assert(accessSequence != nullptr);
	
	if (dataAccess->_weak != newAccessWeakness) {
		Instrument::upgradedDataAccess(
			(accessSequence->_superAccess != nullptr ? accessSequence->_superAccess->_instrumentationId : Instrument::data_access_id_t()),
			dataAccess->_instrumentationId,
			dataAccess->_type, dataAccess->_weak,
			dataAccess->_type, (dataAccess->_weak && newAccessWeakness),
			false, task->getInstrumentationTaskId()
		);
		dataAccess->_weak &= newAccessWeakness; // In fact, just false
	}
	
	// An identical access
	return true; // Do not count this one
}


bool DataAccess::upgradeSameStrengthAccess(Task *task, DataAccess /* INOUT */ *dataAccess, DataAccessType newAccessType)
{
	assert(dataAccess != nullptr);
	
	DataAccessSequence *accessSequence = dataAccess->_dataAccessSequence;
	assert(accessSequence != nullptr);
	
	if (dataAccess->_type == READWRITE_ACCESS_TYPE) {
		// The old access is as restrictive as possible
		return true;
	} else if (dataAccess->_type == WRITE_ACCESS_TYPE) {
		// A write that becomes readwrite
		assert((newAccessType == READWRITE_ACCESS_TYPE) || (newAccessType == READ_ACCESS_TYPE));
		Instrument::upgradedDataAccess(
			(accessSequence->_superAccess != nullptr ? accessSequence->_superAccess->_instrumentationId : Instrument::data_access_id_t()),
			dataAccess->_instrumentationId,
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
			Instrument::removeTaskFromAccessGroup(accessSequence, task->getInstrumentationTaskId());
			Instrument::beginAccessGroup(task->getParent()->getInstrumentationTaskId(), accessSequence, false);
			Instrument::addTaskToAccessGroup(accessSequence, task->getInstrumentationTaskId());
		}
		
		DataAccessType oldAccessType = dataAccess->_type;
		
		// Upgrade the access type
		dataAccess->_type = READWRITE_ACCESS_TYPE;
		
		if (dataAccess->_satisfied) {
			// Calculate if the satisfiability of the upgraded access
			DataAccess *effectivePrevious = accessSequence->getEffectivePrevious(dataAccess);
			bool satisfied = evaluateSatisfiability(effectivePrevious, dataAccess->_type);
			dataAccess->_satisfied = satisfied;
			
			Instrument::upgradedDataAccess(
				(accessSequence->_superAccess != nullptr ? accessSequence->_superAccess->_instrumentationId : Instrument::data_access_id_t()),
				dataAccess->_instrumentationId,
				oldAccessType, dataAccess->_weak,
				newAccessType, dataAccess->_weak,
				!satisfied, task->getInstrumentationTaskId()
			);
			
			return satisfied; // A new chance for the access to not be satisfied
		} else {
			Instrument::upgradedDataAccess(
				(accessSequence->_superAccess != nullptr ? accessSequence->_superAccess->_instrumentationId : Instrument::data_access_id_t()),
				dataAccess->_instrumentationId,
				dataAccess->_type, dataAccess->_weak,
				newAccessType, dataAccess->_weak,
				false, task->getInstrumentationTaskId()
			);
			
			return true; // The old access has already been counted
		}
	}
}


bool DataAccess::upgradeStrongAccessWithWeak(Task *task, DataAccess /* INOUT */ * /* INOUT */ &dataAccess, DataAccessType newAccessType)
{
	assert(dataAccess != nullptr);
	
	DataAccessSequence *accessSequence = dataAccess->_dataAccessSequence;
	assert(accessSequence != nullptr);
	
	if (dataAccess->_type == READWRITE_ACCESS_TYPE) {
		// The old access is as restrictive as possible
		return true;
	} else if (dataAccess->_type == WRITE_ACCESS_TYPE) {
		// A write that becomes readwrite
		assert((newAccessType == READWRITE_ACCESS_TYPE) || (newAccessType == READ_ACCESS_TYPE));
		Instrument::upgradedDataAccess(
			(accessSequence->_superAccess != nullptr ? accessSequence->_superAccess->_instrumentationId : Instrument::data_access_id_t()),
			dataAccess->_instrumentationId,
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
			
			Instrument::data_access_id_t newDataAccessInstrumentationId =
			Instrument::createdDataAccess(
				(accessSequence->_superAccess != nullptr ? accessSequence->_superAccess->_instrumentationId : Instrument::data_access_id_t()),
				newAccessType, true,
				satisfied,
				task->getInstrumentationTaskId()
			);
			
			Instrument::linkedDataAccesses(
				dataAccess->_instrumentationId,
				newDataAccessInstrumentationId,
				true /* Direct? */,
				task->getInstrumentationTaskId()
			);
			
			dataAccess = new DataAccess(
				accessSequence,
				newAccessType, true,
				satisfied,
				task, accessSequence->_accessRange,
				newDataAccessInstrumentationId
			);
			accessSequence->_accessSequence.push_back(*dataAccess); // NOTE: It actually does get the pointer
			
			return satisfied;
		}
	}
}


bool DataAccess::upgradeWeakAccessWithStrong(Task *task, DataAccess /* INOUT */ * /* INOUT */ &dataAccess, DataAccessType newAccessType)
{
	assert(dataAccess != nullptr);
	
	DataAccessSequence *accessSequence = dataAccess->_dataAccessSequence;
	assert(accessSequence != nullptr);
	
	if (newAccessType != READ_ACCESS_TYPE) {
		// A new write or readwrite that subsumes a weak access
		assert((dataAccess->_type != WRITE_ACCESS_TYPE) || (newAccessType != WRITE_ACCESS_TYPE)); // Handled elsewhere
		
		DataAccessType oldAccessType = dataAccess->_type;
		
		// Upgrade the access type
		dataAccess->_type = READWRITE_ACCESS_TYPE;
		dataAccess->_weak = false;
		
		if (dataAccess->_satisfied) {
			// Calculate if the satisfiability of the upgraded access
			DataAccess *effectivePrevious = accessSequence->getEffectivePrevious(dataAccess);
			bool satisfied = evaluateSatisfiability(effectivePrevious, dataAccess->_type);
			dataAccess->_satisfied = satisfied;
			
			Instrument::upgradedDataAccess(
				(accessSequence->_superAccess != nullptr ? accessSequence->_superAccess->_instrumentationId : Instrument::data_access_id_t()),
				dataAccess->_instrumentationId,
				oldAccessType, true,
				newAccessType, false,
				!satisfied, task->getInstrumentationTaskId()
			);
			
			return satisfied; // A new chance for the access to not be satisfied
		} else {
			Instrument::upgradedDataAccess(
				(accessSequence->_superAccess != nullptr ? accessSequence->_superAccess->_instrumentationId : Instrument::data_access_id_t()),
				dataAccess->_instrumentationId,
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
			
			Instrument::upgradedDataAccess(
				(accessSequence->_superAccess != nullptr ? accessSequence->_superAccess->_instrumentationId : Instrument::data_access_id_t()),
				dataAccess->_instrumentationId,
				READ_ACCESS_TYPE, true,
				READ_ACCESS_TYPE, false,
				false, task->getInstrumentationTaskId()
			);
			
			return dataAccess->_satisfied; // A new chance for the access to be accounted
		} else {
			// The new "strong" read must come before the old weak write or weak readwrite
			
			DataAccess *effectivePrevious = accessSequence->getEffectivePrevious(dataAccess);
			bool satisfied = evaluateSatisfiability(effectivePrevious, newAccessType);
			
			// We overwrite the old DataAccess object with the "strong" read and create a new DataAccess after it with the old weak access information
			// This simplifies insertion and the instrumentation
			
			// Instrumentation for the upgrade of the existing access to "strong" read
			Instrument::upgradedDataAccess(
				(accessSequence->_superAccess != nullptr ? accessSequence->_superAccess->_instrumentationId : Instrument::data_access_id_t()),
				dataAccess->_instrumentationId,
				dataAccess->_type, true,
				READ_ACCESS_TYPE, false,
				false, task->getInstrumentationTaskId()
			);
			if (dataAccess->_satisfied != satisfied) {
				Instrument::dataAccessBecomesSatisfied(
					(accessSequence->_superAccess != nullptr ? accessSequence->_superAccess->_instrumentationId : Instrument::data_access_id_t()),
					dataAccess->_instrumentationId,
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
			Instrument::createdDataAccess(
				(accessSequence->_superAccess != nullptr ? accessSequence->_superAccess->_instrumentationId : Instrument::data_access_id_t()),
				oldAccessType, true,
				false,
				task->getInstrumentationTaskId()
			);
			
			Instrument::linkedDataAccesses(
				effectivePrevious->_instrumentationId,
				newDataAccessInstrumentationId,
				!accessSequence->_accessSequence.empty() /* Direct? */,
				task->getInstrumentationTaskId()
			);
			
			dataAccess = new DataAccess(
				accessSequence,
				oldAccessType, true,
				false,
				task, accessSequence->_accessRange,
				newDataAccessInstrumentationId
			);
			accessSequence->_accessSequence.push_back(*dataAccess); // NOTE: It actually does get the pointer
			
			return satisfied;
		}
	}
}


bool DataAccess::upgradeAccess(Task *task, DataAccess /* INOUT */ * /* INOUT */ &dataAccess, DataAccessType newAccessType, bool newAccessWeakness)
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



#endif // DATA_ACCESS_IMPLEMENTATION_HPP
