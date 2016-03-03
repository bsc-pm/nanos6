#ifndef INSTRUMENT_NULL_DEPENDENCIES_BY_ACCESS_LINK_HPP
#define INSTRUMENT_NULL_DEPENDENCIES_BY_ACCESS_LINK_HPP


#include "../InstrumentDependenciesByAccessLinks.hpp"


namespace Instrument {
	inline data_access_id_t createdDataAccess(
		__attribute__((unused)) data_access_id_t superAccessId,
		__attribute__((unused)) DataAccessType accessType,
		__attribute__((unused)) bool weak,
		__attribute__((unused)) DataAccessRange range,
		__attribute__((unused)) bool satisfied,
		__attribute__((unused)) task_id_t originatorTaskId
	) {
		return data_access_id_t();
	}
	
	inline void upgradedDataAccess(
		__attribute__((unused)) data_access_id_t superAccessId,
		__attribute__((unused)) data_access_id_t dataAccessId,
		__attribute__((unused)) DataAccessType previousAccessType,
		__attribute__((unused)) bool previousWeakness,
		__attribute__((unused)) DataAccessType newAccessType,
		__attribute__((unused)) bool newWeakness,
		__attribute__((unused)) bool becomesUnsatisfied,
		__attribute__((unused)) task_id_t triggererTaskId
	) {
	}
	
	inline void dataAccessBecomesSatisfied(
		__attribute__((unused)) data_access_id_t superAccessId,
		__attribute__((unused)) data_access_id_t dataAccessId,
		__attribute__((unused)) task_id_t triggererTaskId,
		__attribute__((unused)) task_id_t targetTaskId
	) {
	}
	
	inline void removedDataAccess(
		__attribute__((unused)) data_access_id_t superAccessId,
		__attribute__((unused)) data_access_id_t dataAccessId,
		__attribute__((unused)) task_id_t triggererTaskId
	) {
	}
	
	inline void linkedDataAccesses(
		__attribute__((unused)) data_access_id_t sourceAccessId,
		__attribute__((unused)) data_access_id_t sinkAccessId,
		__attribute__((unused)) bool direct,
		__attribute__((unused)) task_id_t triggererTaskId
	) {
	}
	
	inline void unlinkedDataAccesses(
		__attribute__((unused)) data_access_id_t sourceAccessId,
		__attribute__((unused)) data_access_id_t sinkAccessId,
		__attribute__((unused)) bool direct,
		__attribute__((unused)) task_id_t triggererTaskId
	) {
	}
	
	inline void reparentedDataAccess(
		__attribute__((unused)) data_access_id_t oldSuperAccessId,
		__attribute__((unused)) data_access_id_t newSuperAccessId,
		__attribute__((unused)) data_access_id_t dataAccessId,
		__attribute__((unused)) task_id_t triggererTaskId
	) {
	}
}


#endif // INSTRUMENT_NULL_DEPENDENCIES_BY_ACCESS_LINK_HPP
