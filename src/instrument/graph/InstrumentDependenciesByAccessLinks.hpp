#ifndef INSTRUMENT_GRAPH_DEPENDENCIES_BY_ACCESS_LINK_HPP
#define INSTRUMENT_GRAPH_DEPENDENCIES_BY_ACCESS_LINK_HPP


#include "InstrumentDataAccessId.hpp"

#include "../api/InstrumentDependenciesByAccessLinks.hpp"


namespace Instrument {
	data_access_id_t createdDataAccess(
		data_access_id_t superAccessId,
		DataAccessType accessType, bool weak, DataAccessRange range,
		bool readSatisfied, bool writeSatisfied, bool globallySatisfied,
		task_id_t originatorTaskId
	);
	
	void upgradedDataAccess(
		data_access_id_t dataAccessId,
		DataAccessType previousAccessType,
		bool previousWeakness,
		DataAccessType newAccessType,
		bool newWeakness,
		bool becomesUnsatisfied,
		task_id_t triggererTaskId
	);
	
	void dataAccessBecomesSatisfied(
		data_access_id_t dataAccessId,
		bool readSatisfied, bool writeSatisfied, bool globallySatisfied,
		task_id_t triggererTaskId,
		task_id_t targetTaskId
	);
	
	void modifiedDataAccessRange(
		data_access_id_t dataAccessId,
		DataAccessRange newRange,
		task_id_t triggererTaskId
	);
	
	data_access_id_t fragmentedDataAccess(
		data_access_id_t dataAccessId,
		task_id_t triggererTaskId
	);
	
	data_access_id_t createdDataSubaccessFragment(
		data_access_id_t dataAccessId,
		task_id_t triggererTaskId
	);
	
	void completedDataAccess(
		data_access_id_t dataAccessId,
		task_id_t triggererTaskId
	);
	
	void dataAccessBecomesRemovable(
		data_access_id_t dataAccessId,
		task_id_t triggererTaskId
	);
	
	void removedDataAccess(
		data_access_id_t dataAccessId,
		task_id_t triggererTaskId
	);
	
	void linkedDataAccesses(
		data_access_id_t sourceAccessId, task_id_t sinkTaskId,
		DataAccessRange range,
		bool direct, bool bidirectional,
		task_id_t triggererTaskId
	);
	
	void unlinkedDataAccesses(
		data_access_id_t sourceAccessId,
		task_id_t sinkTaskId,
		bool direct,
		task_id_t triggererTaskId
	);
	
	void reparentedDataAccess(
		data_access_id_t oldSuperAccessId,
		data_access_id_t newSuperAccessId,
		data_access_id_t dataAccessId,
		task_id_t triggererTaskId
	);
	
}


#endif // INSTRUMENT_GRAPH_DEPENDENCIES_BY_ACCESS_LINK_HPP
