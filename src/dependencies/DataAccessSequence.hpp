#ifndef DATA_ACCESS_SEQUENCE_HPP
#define DATA_ACCESS_SEQUENCE_HPP

#include <boost/intrusive/list.hpp>

#include "DataAccess.hpp"
#include "DataAccessSequenceLinkingArtifacts.hpp"
#include "DataAccessType.hpp"
#include "lowlevel/SpinLock.hpp"

#include <InstrumentDataAccessSequenceId.hpp>


class Task;


struct DataAccessSequence {
	//! \brief SpinLock to protect the sequence
	SpinLock _lock;
	
	typedef boost::intrusive::list<DataAccess, boost::intrusive::function_hook<DataAccessSequenceLinkingArtifacts>> access_sequence_t;
	
	//! \brief Ordered sequence of accesses to the same location
	access_sequence_t _accessSequence;
	
	//! An identifier for the instrumentation
	Instrument::data_access_sequence_id_t _instrumentationId;
	
	
	inline DataAccessSequence();
	
	
	//! \brief Reevaluate the satisfactibility of a DataAccess according to the once immediately preceeding it (if any)
	//! 
	//! \param[in] position an iterator to the list position of the DataAccess to be reevaluated
	//! 
	//! \returns true if the DataAccess has become satisfied
	inline bool reevaluateSatisfactibility(access_sequence_t::iterator position);
	
	
	//! \brief adds a task access to the sequence taking into account repeated accesses
	//! 
	//! \param[in] task the task that performs the access
	//! \param[in] accessType the type of access
	//! \param[out] dataAccess gets initialized with a pointer to the new DataAccess object or null if there was already a previous one for that task
	//! 
	//! \returns true is the access can be started
	//!
	//! The new DataAccess object has the task as its originator and is inserted in the DataAccessSequence.
	//! However, it is not inserted in the list of accesses of the Task.
	//! 
	//! If the task has already a previous access, it may be upgraded if necessary, and dataAccess is set to null. The return
	//! value indicates if the new access produces an additional dependency (only possible if the previous one did not).
	inline bool addTaskAccess(Task *task, DataAccessType accessType, DataAccess *&dataAccess);
	
};


#include "DataAccessSequenceLinkingArtifactsImplementation.hpp"


#endif // DATA_ACCESS_SEQUENCE_HPP
