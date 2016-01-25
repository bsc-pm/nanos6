#ifndef INSTRUMENT_DEPENDENCIES_BY_ACCESS_SEQUENCE_HPP
#define INSTRUMENT_DEPENDENCIES_BY_ACCESS_SEQUENCE_HPP


#include <InstrumentDataAccessId.hpp>
#include <InstrumentDataAccessSequenceId.hpp>
#include <InstrumentTaskId.hpp>

#include "dependencies/DataAccessType.hpp"


namespace Instrument {
	//! \file
	//! \name Dependency instrumentation by access sequences
	//! @{
	//! This group of functions is useful for instrumenting task dependencies. It passes the actual information that the
	//! runtime manages so that the instrumentation can closely represent its behaviour.
	//! 
	//! The main concepts are the sequence of data accesses and the data access.
	//!
	//! A data access is an access of a task to a storage that can produce dependencies. A sequence of those is a list
	//! of thore ordered in creation time that corresponds to the same storage.
	//! 
	//! As tasks are executed, they make the data accesses of other tasks satisfied, that is, they liberate a dependency
	//! between the task that finishes and the task that originates the satisfied data access. Therefore, when using this
	//! interface, dependencies are determined at the time that they are satisfied.
	//! 
	//! Since this interface depends on the information that the runtime stores, it cannot detect reliably situations in
	//! which there are not enough data accesses in a given sequence.
	//! 
	//! In addition the interface for instrumenting taskwaits can help to differentiate between cases in which a sequence
	//! is depleted due to a taskwait from the cases in which they deplete due to the conditions of the execution
	//! environment (for instance the number of threads).
	
	
	//! \brief Called when a new DataAccessSequence is created
	//! 
	//! \param parentDataAccessId the data access identifier that corresponds to the parent task, or data_access_id_t() if there is no matching parent access
	//! \param triggererTaskId the identifier of the task that triggers the creation of the sequence if any, otherwise task_id_t()
	//! 
	//! \returns an identifier for the DataAccessSequence
	data_access_sequence_id_t registerAccessSequence(data_access_id_t parentDataAccessId, task_id_t triggererTaskId);
	
	//! \brief Called when a new DataAccess is added to a DataAccessSequence
	//! 
	//! \param dataAccessSequenceId the identifier of the sequence returned by Instrument::newAccessSequence when it was created
	//! \param accessType the type of access of the new DataAccess
	//! \param satisfied whether the access does not preclude the task from running immediately
	//! \param originatorTaskId the identifier of the task that will perform the access as returned in the call to Instrument::enterAddTask
	//! 
	//! \returns an identifier for the new data access
	data_access_id_t addedDataAccessInSequence(data_access_sequence_id_t dataAccessSequenceId, DataAccessType accessType, bool satisfied, task_id_t originatorTaskId);
	
	//! \brief Called when a DataAccess has its type of access upgraded
	//! 
	//! Note that this function may be called with previousAccessType == newAccessType in case of a repeated access
	//! 
	//! \param dataAccessSequenceId the identifier of the sequence returned by Instrument::newAccessSequence when it was created
	//! \param dataAccessId the identifier of the DataAccess returned in the previous call to Instrument::registerDataAccessInSequence
	//! \param previousAccessType the type of access that will be upgraded
	//! \param newAccessType the type of access to which it will be upgraded
	//! \param becomesUnsatisfied indicates if the DataAccess was satisfied and has become unsatisfied as a result of the upgrade
	//! \param triggererTaskId the identifier of the task that trigers the change
	//! 
	void upgradedDataAccessInSequence(data_access_sequence_id_t dataAccessSequenceId, data_access_id_t dataAccessId, DataAccessType previousAccessType, DataAccessType newAccessType, bool becomesUnsatisfied, task_id_t triggererTaskId);
	
	//! \brief Called when a DataAccess becomes satisfied
	//! 
	//! \param dataAccessSequenceId the identifier of the sequence returned by Instrument::newAccessSequence when it was created
	//! \param dataAccessId the identifier of the DataAccess that becomes satisfied as returned in the previous call to Instrument::registerDataAccessInSequence
	//! \param triggererTaskId the identifier of the task that trigers the change
	//! \param targetTaskId the identifier of the task that will perform the now satisfied DataAccess
	void dataAccessBecomesSatisfied(data_access_sequence_id_t dataAccessSequenceId, data_access_id_t dataAccessId, task_id_t triggererTaskId, task_id_t targetTaskId);
	
	//! \brief Called when a DataAccess has been removed from a sequence
	//! 
	//! \param dataAccessSequenceId the identifier of the sequence returned by Instrument::newAccessSequence when it was created
	//! \param dataAccessId the identifier of the DataAccess as returned in the previous call to Instrument::registerDataAccessInSequence
	//! \param triggererTaskId the identifier of the task that trigers the change
	void removedDataAccessFromSequence(data_access_sequence_id_t dataAccessSequenceId, data_access_id_t dataAccessId, task_id_t triggererTaskId);
	
	//! \brief Called when a DataAccess has is moved from one DataAccessSequence to another
	//! 
	//! \param previousDataAccessSequenceId the identifier of the sequence from which the DataAccess is removed
	//! \param newDataAccessSequenceId the identifier of the sequence to which the DataAccess is inserted
	//! \param dataAccessId the identifier of the DataAccess that is moved
	//! \param beforeDataAccessId the identifier of the DataAccess from the target DataAccessSequence on fron of which the access will be placed
	//! \param triggererTaskId the identifier of the task that trigers the change
	//! 
	void replacedSequenceOfDataAccess(data_access_sequence_id_t previousDataAccessSequenceId, data_access_sequence_id_t newDataAccessSequenceId, data_access_id_t dataAccessId, data_access_id_t beforeDataAccessId, task_id_t triggererTaskId);
	
	//! @}
}


#endif // INSTRUMENT_DEPENDENCIES_BY_ACCESS_SEQUENCE_HPP
