/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_IMPLEMENTATION_HPP
#define DATA_ACCESS_IMPLEMENTATION_HPP

#include <cassert>
#include <mutex>

#include "DataAccess.hpp"
#include "LinearRegionDataAccessMapImplementation.hpp"
#include "tasks/Task.hpp"

#include <InstrumentDependenciesByAccessLinks.hpp>


template <typename EffectivePreviousProcessorType>
bool DataAccess::processEffectivePrevious(
	DataAccessRange const &range,
	bool processDirectPrevious,
	EffectivePreviousProcessorType effectivePreviousProcessor
) {
	if (_previous.empty()) {
		if (_superAccess != nullptr) {
			return _superAccess->processEffectivePrevious(range, true, effectivePreviousProcessor);
		} else {
			return true;
		}
	} else {
		return _previous.processIntersectingAndMissing(
			range,
			[&](DataAccessPreviousLinks::iterator previousPosition) -> bool {
				if (processDirectPrevious) {
					return effectivePreviousProcessor(previousPosition);
				} else {
					return true;
				}
			},
			[&](DataAccessRange const &missingRange) -> bool {
				if (_superAccess != nullptr) {
					return _superAccess->processEffectivePrevious(missingRange, true, effectivePreviousProcessor);
				}
				return true;
			}
		);
	}
}


inline void DataAccess::fullLinkTo(DataAccessRange const &range, DataAccess *target, bool blocker, Instrument::task_id_t triggererTaskInstrumentationId)
{
	assert(target != nullptr);
	assert(target->_originator != nullptr);
	
	Instrument::linkedDataAccesses(
		_instrumentationId, target->_originator->getInstrumentationTaskId(),
		range,
		/* direct */ true,
		/* bidirectional */ true,
		triggererTaskInstrumentationId
	);
	
	_next.insert(DataAccessNextLinkContents(range, target, !blocker));
	target->_previous.insert(LinearRegionDataAccessMapNode(range, this));
}


inline bool DataAccess::evaluateSatisfiability(DataAccess *effectivePrevious, DataAccessType nextAccessType)
{
	if (effectivePrevious == nullptr) {
		// The first position is satisfied
		return true;
	}
	
	if (effectivePrevious->_blockerCount != 0) {
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
	assert(effectivePrevious->_blockerCount == 0);
	if (effectivePrevious->_type == READ_ACCESS_TYPE) {
		// Consecutive reads are satisfied together
		return true;
	} else {
		assert((effectivePrevious->_type == WRITE_ACCESS_TYPE) || (effectivePrevious->_type == READWRITE_ACCESS_TYPE));
		// Read after Write
		return false;
	}
}


inline bool DataAccess::propagatesSatisfiability(DataAccessType nextAccessType)
{
	return (_type == READ_ACCESS_TYPE) && (nextAccessType == READ_ACCESS_TYPE);
}


bool DataAccess::upgradeSameTypeAccess(Task *task, DataAccess /* INOUT */ *dataAccess, bool newAccessWeakness)
{
	assert(dataAccess != nullptr);
	
	if (dataAccess->_weak != newAccessWeakness) {
		Instrument::upgradedDataAccess(
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


bool DataAccess::updateBlockerCountAndLinkSatisfiability()
{
	bool wasSatisfied = (_blockerCount == 0);
	
	_blockerCount = 0;
	
	// Process the directly linked previous accesses
	for (DataAccessPreviousLinks::iterator it = _previous.begin(); it != _previous.end(); it++) {
		LinearRegionDataAccessMapNode const &previousLink = *it;
		
		DataAccess *previous = previousLink._access;
		assert(previous != nullptr);
		
		// Update the blocker count
		bool satisfied = evaluateSatisfiability(previous, _type);
		if (!satisfied) {
			_blockerCount++;
		}
		
		// Update the satisfiability on the link from the previous to "this"
		DataAccessRange const &range = previousLink.getAccessRange();
		DataAccessNextLinks::iterator forwardLinkPosition = previous->_next.find(range);
		assert(forwardLinkPosition != previous->_next.end());
		assert(forwardLinkPosition->_access == this);
		forwardLinkPosition->_satisfied = satisfied;
	}
	
	// Count the non-directly linked effective previous blockers
	processEffectivePrevious(
		_range,
		false,
		[&](DataAccessPreviousLinks::iterator effectivePreviousPosition) -> bool {
			bool satisfied = evaluateSatisfiability(effectivePreviousPosition->_access, _type);
			if (!satisfied) {
				_blockerCount++;
			}
			
			return true;
		}
	);
	
	bool isSatisfied = (_blockerCount == 0);
	
	return (wasSatisfied != isSatisfied);
}


int DataAccess::calculateBlockerCount(DataAccessType accessType)
{
	int result = 0;
	
	for (DataAccessPreviousLinks::iterator it = _previous.begin(); it != _previous.end(); it++) {
		bool satisfied = evaluateSatisfiability(it->_access, accessType);
		if (!satisfied) {
			result++;
		}
	}
	
	processEffectivePrevious(
		_range,
		false,
		[&](DataAccessPreviousLinks::iterator effectivePreviousPosition) -> bool {
			bool satisfied = evaluateSatisfiability(effectivePreviousPosition->_access, accessType);
			if (!satisfied) {
				result++;
			}
			
			return true;
		}
	);
	
	return result;
}


bool DataAccess::updateBlockerCount(DataAccessType accessType)
{
	bool wasSatisfied = (_blockerCount == 0);
	_blockerCount = calculateBlockerCount(accessType);
	bool isSatisfied = (_blockerCount == 0);
	
	return (wasSatisfied != isSatisfied);
}


bool DataAccess::upgradeSameStrengthAccess(Task *task, DataAccess /* INOUT */ *dataAccess, DataAccessType newAccessType)
{
	assert(dataAccess != nullptr);
	assert(task != nullptr);
	
	Instrument::task_id_t taskInstrumentationId = task->getInstrumentationTaskId();
	
	Instrument::data_access_id_t superAccessId = (dataAccess->_superAccess != nullptr ? dataAccess->_superAccess->_instrumentationId : Instrument::data_access_id_t());
	if (dataAccess->_type == READWRITE_ACCESS_TYPE) {
		// The old access is as restrictive as possible
		return true;
	} else if (dataAccess->_type == WRITE_ACCESS_TYPE) {
		// A write that becomes readwrite
		assert((newAccessType == READWRITE_ACCESS_TYPE) || (newAccessType == READ_ACCESS_TYPE));
		Instrument::upgradedDataAccess(
			dataAccess->_instrumentationId,
			dataAccess->_type, dataAccess->_weak,
			READWRITE_ACCESS_TYPE, dataAccess->_weak,
			false, taskInstrumentationId
		);
		dataAccess->_type = newAccessType;
		
		// The essential type of access did not change, and thus neither did its satisfiability
		return true; // Do not count this one
	} else {
		// Upgrade a read into a write or a readwrite
		assert(dataAccess->_type == READ_ACCESS_TYPE);
		assert((newAccessType == WRITE_ACCESS_TYPE) || (newAccessType == READWRITE_ACCESS_TYPE));
		
		DataAccessType oldAccessType = dataAccess->_type;
		
		// Upgrade the access type
		dataAccess->_type = newAccessType;
		
		bool becomesUnsatisfied = dataAccess->updateBlockerCountAndLinkSatisfiability();
		
		Instrument::upgradedDataAccess(
			dataAccess->_instrumentationId,
			oldAccessType, dataAccess->_weak,
			newAccessType, dataAccess->_weak,
			(dataAccess->_blockerCount != 0), taskInstrumentationId
		);
		
		return !becomesUnsatisfied;
	}
}


bool DataAccess::upgradeStrongAccessWithWeak(Task *task, DataAccess /* INOUT */ * /* INOUT */ &dataAccess, DataAccessType newAccessType)
{
	assert(dataAccess != nullptr);
	assert(task != nullptr);
	
	Instrument::task_id_t taskInstrumentationId = task->getInstrumentationTaskId();
	
	Instrument::data_access_id_t superAccessId = (dataAccess->_superAccess != nullptr ? dataAccess->_superAccess->_instrumentationId : Instrument::data_access_id_t());
	if (dataAccess->_type == READWRITE_ACCESS_TYPE) {
		// The old access is as restrictive as possible
		return true;
	} else if (dataAccess->_type == WRITE_ACCESS_TYPE) {
		// A write that becomes readwrite
		assert((newAccessType == READWRITE_ACCESS_TYPE) || (newAccessType == READ_ACCESS_TYPE));
		Instrument::upgradedDataAccess(
			dataAccess->_instrumentationId,
			dataAccess->_type, false,
			READWRITE_ACCESS_TYPE, false,
			false, taskInstrumentationId
		);
		dataAccess->_type = READWRITE_ACCESS_TYPE;
		
		// The essential type of access did not change, and thus neither did its satisfiability
		return true; // Do not count this one
	} else {
		assert(dataAccess->_type == READ_ACCESS_TYPE);
		
		if (newAccessType == READ_ACCESS_TYPE) {
			return true;
		} else {
			DataAccess *oldDataAccess = dataAccess;
			
			Instrument::data_access_id_t newDataAccessInstrumentationId = Instrument::createdDataAccess(
				superAccessId,
				newAccessType, true,
				dataAccess->_range,
				false, false, false, // Not satisfied
				taskInstrumentationId
			);
			
			dataAccess = new DataAccess(
				oldDataAccess->_superAccess, oldDataAccess->_lock, oldDataAccess->_bottomMap,
				newAccessType, true,
				1, /* Blocked by the previous "strong" read */
				task, dataAccess->_range,
				newDataAccessInstrumentationId
			);
			
			assert(oldDataAccess->_next.empty());
			oldDataAccess->fullLinkTo(dataAccess->_range, dataAccess, true /* Blocking */, taskInstrumentationId);
			
			return true; // Do not count it since it is a weak access
		}
	}
}


bool DataAccess::upgradeWeakAccessWithStrong(Task *task, DataAccess /* INOUT */ * /* INOUT */ &dataAccess, DataAccessType newAccessType)
{
	assert(dataAccess != nullptr);
	assert(task != nullptr);
	
	Instrument::task_id_t taskInstrumentationId = task->getInstrumentationTaskId();
	
	Instrument::data_access_id_t superAccessId = (dataAccess->_superAccess != nullptr ? dataAccess->_superAccess->_instrumentationId : Instrument::data_access_id_t());
	if (newAccessType != READ_ACCESS_TYPE) {
		// A new write or readwrite that subsumes a weak access
		assert((dataAccess->_type != WRITE_ACCESS_TYPE) || (newAccessType != WRITE_ACCESS_TYPE)); // Handled elsewhere
		
		DataAccessType oldAccessType = dataAccess->_type;
		
		// Upgrade the access type
		dataAccess->_type = READWRITE_ACCESS_TYPE;
		dataAccess->_weak = false;
		
		Instrument::upgradedDataAccess(
			dataAccess->_instrumentationId,
			oldAccessType, true,
			newAccessType, false,
			(dataAccess->_blockerCount != 0), taskInstrumentationId
		);
		
		return (dataAccess->_blockerCount == 0); // If the "stong" access is blocked, then count it, since the "weak" one did not count
	} else {
		// A new "strong" read to be combined with an already existing weak access
		assert(newAccessType == READ_ACCESS_TYPE);
		
		if (dataAccess->_type == READ_ACCESS_TYPE) {
			dataAccess->_weak = false;
			
			Instrument::upgradedDataAccess(
				dataAccess->_instrumentationId,
				READ_ACCESS_TYPE, true,
				READ_ACCESS_TYPE, false,
				(dataAccess->_blockerCount != 0), taskInstrumentationId
			);
			
			return (dataAccess->_blockerCount == 0); // A new chance for the access to be accounted
		} else {
			// The new "strong" read must come before the old weak write or weak readwrite
			
			// We overwrite the old DataAccess object with the "strong" read and create a new DataAccess after it with the old weak access information
			// This simplifies insertion and the instrumentation
			
			DataAccessType oldAccessType = dataAccess->_type;
			
			// Update existing access to "strong" read
			dataAccess->_type = READ_ACCESS_TYPE;
			dataAccess->_weak = false;
			dataAccess->_blockerCount = 0;
			
			dataAccess->updateBlockerCountAndLinkSatisfiability();
			
			// Instrumentation for the upgrade of the existing access to "strong" read
			Instrument::upgradedDataAccess(
				dataAccess->_instrumentationId,
				oldAccessType, true,
				READ_ACCESS_TYPE, false,
				(dataAccess->_blockerCount == 0), taskInstrumentationId
			);
			
			// New object with the old information
			Instrument::data_access_id_t newDataAccessInstrumentationId =
			Instrument::createdDataAccess(
				superAccessId,
				oldAccessType, true,
				dataAccess->_range,
				false, false, false /* Not satisfied */,
				taskInstrumentationId
			);
			
			DataAccess *oldDataAccess = dataAccess;
			
			dataAccess = new DataAccess(
				oldDataAccess->_superAccess, oldDataAccess->_lock, oldDataAccess->_bottomMap,
				oldAccessType, true,
				1, /* Blocked by the previous "strong" read */
				task, dataAccess->_range,
				newDataAccessInstrumentationId
			);
			
			assert(oldDataAccess->_next.empty());
			oldDataAccess->fullLinkTo(oldDataAccess->_range, dataAccess, true, taskInstrumentationId);
			
			return (oldDataAccess->_blockerCount == 0);
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
