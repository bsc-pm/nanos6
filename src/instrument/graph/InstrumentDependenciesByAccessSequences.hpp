#ifndef INSTRUMENT_GRAPH_DEPENDENCIES_BY_ACCESS_SEQUENCE_HPP
#define INSTRUMENT_GRAPH_DEPENDENCIES_BY_ACCESS_SEQUENCE_HPP


#include "InstrumentDataAccessId.hpp"
#include "InstrumentDataAccessSequenceId.hpp"

#include "../InstrumentDependenciesByAccessSequences.hpp"


namespace Instrument {
	data_access_sequence_id_t registerAccessSequence(data_access_id_t parentDataAccessId, task_id_t triggererTaskId);
	
	data_access_id_t addedDataAccessInSequence(data_access_sequence_id_t dataAccessSequenceId, DataAccessType accessType, bool weak, bool satisfied, task_id_t originatorTaskId);
	
	void upgradedDataAccessInSequence(data_access_sequence_id_t dataAccessSequenceId, data_access_id_t dataAccessId, DataAccessType previousAccessType, bool previousWeakness, DataAccessType newAccessType, bool newWeakness, bool becomesUnsatisfied, task_id_t originatorTaskId);
	
	void dataAccessBecomesSatisfied(data_access_sequence_id_t dataAccessSequenceId, data_access_id_t dataAccessId, task_id_t triggererTaskId, task_id_t targetTaskId);
	
	void removedDataAccessFromSequence(data_access_sequence_id_t dataAccessSequenceId, data_access_id_t dataAccessId, task_id_t triggererTaskId);
	
	void replacedSequenceOfDataAccess(data_access_sequence_id_t previousDataAccessSequenceId, data_access_sequence_id_t newDataAccessSequenceId, data_access_id_t dataAccessId, data_access_id_t beforeDataAccessId, task_id_t triggererTaskId);
}


#endif // INSTRUMENT_GRAPH_DEPENDENCIES_BY_ACCESS_SEQUENCE_HPP
