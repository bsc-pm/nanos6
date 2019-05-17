/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_DEPENDENCIES_BY_ACCESS_LINK_HPP
#define INSTRUMENT_EXTRAE_DEPENDENCIES_BY_ACCESS_LINK_HPP


#include "InstrumentDataAccessId.hpp"
#include "InstrumentExtrae.hpp"
#include "InstrumentTaskId.hpp"
#include "../api/InstrumentDependenciesByAccessLinks.hpp"

#include <InstrumentInstrumentationContext.hpp>


namespace Instrument {
	inline data_access_id_t createdDataAccess(
		__attribute__((unused)) data_access_id_t *superAccessId,
		__attribute__((unused)) DataAccessType accessType,
		bool weak,
		__attribute__((unused)) DataAccessRegion region,
		__attribute__((unused)) bool readSatisfied,
		__attribute__((unused)) bool writeSatisfied,
		__attribute__((unused)) bool globallySatisfied,
		__attribute__((unused)) access_object_type_t objectType,
		__attribute__((unused)) task_id_t originatorTaskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		return data_access_id_t(accessType, weak, objectType, originatorTaskId);
	}
	
	inline void upgradedDataAccess(
		__attribute__((unused)) data_access_id_t &dataAccessId,
		__attribute__((unused)) DataAccessType previousAccessType,
		__attribute__((unused)) bool previousWeakness,
		__attribute__((unused)) DataAccessType newAccessType,
		__attribute__((unused)) bool newWeakness,
		__attribute__((unused)) bool becomesUnsatisfied,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void dataAccessBecomesSatisfied(
		data_access_id_t &dataAccessId,
		__attribute__((unused)) bool globallySatisfied,
		task_id_t targetTaskId,
		InstrumentationContext const &context
	) {
		if ((_detailLevel < 1) || (dataAccessId._originator._taskInfo == nullptr)) {
			return;
		}
		
		assert(!globallySatisfied || (dataAccessId._originator == targetTaskId));
		
		if ((context._taskId == targetTaskId) || (context._taskId._taskInfo == nullptr)) {
			return;
		}
		
		if (context._taskId._taskInfo->_parent == context._taskId._taskInfo) {
			// The access is ready during its instantiation
			return;
		}
		
		if ((dataAccessId._objectType != regular_access_type) && (dataAccessId._objectType != taskwait_type)) {
			return;
		}
		
		// Taskwait control dependencies are only emitted when the detail level is at least 8
		if ((dataAccessId._objectType == taskwait_type) && (_detailLevel < 8)) {
			return;
		}
		
		dependency_tag_t emitTag;
		if (dataAccessId._objectType == taskwait_type) {
			emitTag = control_dependency_tag;
		} else {
			assert(dataAccessId._objectType == regular_access_type);
			if (dataAccessId._weak) {
				emitTag = weak_data_dependency_tag;
			} else {
				emitTag = strong_data_dependency_tag;
			}
		}
		
		Extrae::predecessor_entry_t entry(context._taskId._taskInfo->_taskId, emitTag);
		
		// For now we avoid weak accesses, since the task may already have started
		bool mustEmit = false;
		if (!dataAccessId._weak) {
			targetTaskId._taskInfo->_lock.lock();
			auto it = targetTaskId._taskInfo->_predecessors.find(entry);
			if (it == targetTaskId._taskInfo->_predecessors.end()) {
				targetTaskId._taskInfo->_predecessors.insert(entry);
				mustEmit = true;
			}
			targetTaskId._taskInfo->_lock.unlock();
		}
		
		if (!mustEmit) {
			return;
		}
		
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 0;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 0;
		ce.nCommunications = 1;
		
		ce.Communications = (extrae_user_communication_t *) alloca(sizeof(extrae_user_communication_t) * ce.nCommunications);
		
		ce.Communications[0].type = EXTRAE_USER_SEND;
		ce.Communications[0].tag = (extrae_comm_tag_t) emitTag;
		ce.Communications[0].size = (context._taskId._taskInfo->_taskId << 32) + targetTaskId._taskInfo->_taskId;
		ce.Communications[0].partner = EXTRAE_COMM_PARTNER_MYSELF;
		ce.Communications[0].id = (context._taskId._taskInfo->_taskId << 32) + targetTaskId._taskInfo->_taskId;
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	
	
	inline void modifiedDataAccessRegion(
		__attribute__((unused)) data_access_id_t &dataAccessId,
		__attribute__((unused)) DataAccessRegion newRegion,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline data_access_id_t fragmentedDataAccess(
		data_access_id_t &dataAccessId,
		__attribute__((unused)) DataAccessRegion newRegion,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		return data_access_id_t(dataAccessId);
	}
	
	inline data_access_id_t createdDataSubaccessFragment(
		__attribute__((unused)) data_access_id_t &dataAccessId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		return data_access_id_t();
	}
	
	inline void completedDataAccess(
		__attribute__((unused)) data_access_id_t &dataAccessId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void dataAccessBecomesRemovable(
		__attribute__((unused)) data_access_id_t &dataAccessId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void removedDataAccess(
		__attribute__((unused)) data_access_id_t &dataAccessId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void linkedDataAccesses(
		__attribute__((unused)) data_access_id_t &sourceAccessId,
		__attribute__((unused)) task_id_t sinkTaskId,
		__attribute__((unused)) access_object_type_t sinkObjectType,
		__attribute__((unused)) DataAccessRegion region,
		__attribute__((unused)) bool direct,
		__attribute__((unused)) bool bidirectional,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void unlinkedDataAccesses(
		__attribute__((unused)) data_access_id_t &sourceAccessId,
		__attribute__((unused)) task_id_t sinkTaskId,
		__attribute__((unused)) access_object_type_t sinkObjectType,
		__attribute__((unused)) bool direct,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void reparentedDataAccess(
		__attribute__((unused)) data_access_id_t &oldSuperAccessId,
		__attribute__((unused)) data_access_id_t &newSuperAccessId,
		__attribute__((unused)) data_access_id_t &dataAccessId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void newDataAccessProperty(
		__attribute__((unused)) data_access_id_t &dataAccessId,
		__attribute__((unused)) char const *shortPropertyName,
		__attribute__((unused)) char const *longPropertyName,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void newDataAccessLocation(
		__attribute__((unused)) data_access_id_t &dataAccessId,
		__attribute__((unused)) MemoryPlace const *newLocation,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
}


#endif // INSTRUMENT_EXTRAE_DEPENDENCIES_BY_ACCESS_LINK_HPP
