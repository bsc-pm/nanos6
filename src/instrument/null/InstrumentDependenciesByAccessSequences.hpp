#ifndef INSTRUMENT_NULL_DEPENDENCIES_BY_ACCESS_SEQUENCE_HPP
#define INSTRUMENT_NULL_DEPENDENCIES_BY_ACCESS_SEQUENCE_HPP


#include "../InstrumentDependenciesByAccessSequences.hpp"


namespace Instrument {
	inline data_access_sequence_id_t registerAccessSequence()
	{
		return data_access_sequence_id_t();
	}
	
	inline data_access_id_t addedDataAccessInSequence(
		__attribute__((unused)) data_access_sequence_id_t dataAccessSequenceId,
		__attribute__((unused)) DataAccessType accessType,
		__attribute__((unused)) bool satisfied,
		__attribute__((unused)) task_id_t originatorTaskId
	) {
		return data_access_id_t();
	}
	
	inline void upgradedDataAccessInSequence(
		__attribute__((unused)) data_access_sequence_id_t dataAccessSequenceId,
		__attribute__((unused)) data_access_id_t dataAccessId,
		__attribute__((unused)) DataAccessType previousAccessType,
		__attribute__((unused)) DataAccessType newAccessType,
		__attribute__((unused)) bool becomesUnsatisfied,
		__attribute__((unused)) task_id_t triggererTaskId
	) {
	}
	
	inline void dataAccessBecomesSatisfied(
		__attribute__((unused)) data_access_sequence_id_t dataAccessSequenceId,
		__attribute__((unused)) data_access_id_t dataAccessId,
		__attribute__((unused)) task_id_t triggererTaskId,
		__attribute__((unused)) task_id_t targetTaskId
	) {
	}
	
	inline void removedDataAccessFromSequence(
		__attribute__((unused)) data_access_sequence_id_t dataAccessSequenceId,
		__attribute__((unused)) data_access_id_t dataAccessId,
		__attribute__((unused)) task_id_t triggererTaskId
	) {
	}
	
}


#endif // INSTRUMENT_NULL_DEPENDENCIES_BY_ACCESS_SEQUENCE_HPP
