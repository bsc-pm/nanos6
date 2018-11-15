/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_DEPENDENCIES_BY_ACCESS_LINK_HPP
#define INSTRUMENT_NULL_DEPENDENCIES_BY_ACCESS_LINK_HPP


#include "../api/InstrumentDependenciesByAccessLinks.hpp"


namespace Instrument {
	inline data_access_id_t createdDataAccess(
		__attribute__((unused)) data_access_id_t *superAccessId,
		__attribute__((unused)) DataAccessType accessType,
		__attribute__((unused)) bool weak,
		__attribute__((unused)) DataAccessRegion region,
		__attribute__((unused)) bool readSatisfied,
		__attribute__((unused)) bool writeSatisfied,
		__attribute__((unused)) bool globallySatisfied,
		__attribute__((unused)) access_object_type_t objectType,
		__attribute__((unused)) task_id_t originatorTaskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		return data_access_id_t();
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
		__attribute__((unused)) data_access_id_t &dataAccessId,
		__attribute__((unused)) bool globallySatisfied,
		__attribute__((unused)) task_id_t targetTaskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void modifiedDataAccessRegion(
		__attribute__((unused)) data_access_id_t &dataAccessId,
		__attribute__((unused)) DataAccessRegion newRegion,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline data_access_id_t fragmentedDataAccess(
		__attribute__((unused)) data_access_id_t &dataAccessId,
		__attribute__((unused)) DataAccessRegion newRegion,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		return data_access_id_t();
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


#endif // INSTRUMENT_NULL_DEPENDENCIES_BY_ACCESS_LINK_HPP
