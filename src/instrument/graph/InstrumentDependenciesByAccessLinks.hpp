/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_GRAPH_DEPENDENCIES_BY_ACCESS_LINK_HPP
#define INSTRUMENT_GRAPH_DEPENDENCIES_BY_ACCESS_LINK_HPP


#include "InstrumentDataAccessId.hpp"

#include "../api/InstrumentDependenciesByAccessLinks.hpp"


namespace Instrument {
	data_access_id_t createdDataAccess(
		data_access_id_t superAccessId,
		DataAccessType accessType, bool weak, DataAccessRegion region,
		bool readSatisfied, bool writeSatisfied, bool globallySatisfied,
		access_object_type_t objectType,
		task_id_t originatorTaskId, InstrumentationContext const &context
	);
	
	void upgradedDataAccess(
		data_access_id_t dataAccessId,
		DataAccessType previousAccessType,
		bool previousWeakness,
		DataAccessType newAccessType,
		bool newWeakness,
		bool becomesUnsatisfied,
		InstrumentationContext const &context
	);
	
	void dataAccessBecomesSatisfied(
		data_access_id_t dataAccessId,
		bool readSatisfied, bool writeSatisfied, bool globallySatisfied,
		task_id_t targetTaskId, InstrumentationContext const &context
	);
	
	void modifiedDataAccessRegion(
		data_access_id_t dataAccessId,
		DataAccessRegion newRegion,
		InstrumentationContext const &context
	);
	
	data_access_id_t fragmentedDataAccess(
		data_access_id_t dataAccessId,
		InstrumentationContext const &context
	);
	
	data_access_id_t createdDataSubaccessFragment(
		data_access_id_t dataAccessId,
		InstrumentationContext const &context
	);
	
	void completedDataAccess(
		data_access_id_t dataAccessId,
		InstrumentationContext const &context
	);
	
	void dataAccessBecomesRemovable(
		data_access_id_t dataAccessId,
		InstrumentationContext const &context
	);
	
	void removedDataAccess(
		data_access_id_t dataAccessId,
		InstrumentationContext const &context
	);
	
	void linkedDataAccesses(
		data_access_id_t sourceAccessId, task_id_t sinkTaskId, access_object_type_t sinkObjectType,
		DataAccessRegion region,
		bool direct, bool bidirectional,
		InstrumentationContext const &context
	);
	
	void unlinkedDataAccesses(
		data_access_id_t sourceAccessId,
		task_id_t sinkTaskId, access_object_type_t sinkObjectType,
		bool direct,
		InstrumentationContext const &context
	);
	
	void reparentedDataAccess(
		data_access_id_t oldSuperAccessId,
		data_access_id_t newSuperAccessId,
		data_access_id_t dataAccessId,
		InstrumentationContext const &context
	);
	
	void newDataAccessProperty(
		data_access_id_t dataAccessId,
		char const *shortPropertyName,
		char const *longPropertyName,
		InstrumentationContext const &context
	);

}


#endif // INSTRUMENT_GRAPH_DEPENDENCIES_BY_ACCESS_LINK_HPP
