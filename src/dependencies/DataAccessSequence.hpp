#ifndef DATA_ACCESS_SEQUENCE_HPP
#define DATA_ACCESS_SEQUENCE_HPP

#include <boost/intrusive/list.hpp>

#include "DataAccessRange.hpp"
#include "DataAccessSequenceLinkingArtifacts.hpp"
#include "DataAccessType.hpp"
#include "lowlevel/SpinLock.hpp"

#include <InstrumentDataAccessSequenceId.hpp>


class Task;
struct DataAccess;


struct DataAccessSequence {
	//! The range of data covered by the accesses of this sequence
	DataAccessRange _accessRange;
	
	//! \brief SpinLock to protect the sequence
	SpinLock _lock;
	
	typedef boost::intrusive::list<DataAccess, boost::intrusive::function_hook<DataAccessSequenceLinkingArtifacts>> access_sequence_t;
	
	//! \brief Ordered sequence of accesses to the same location
	access_sequence_t _accessSequence;
	
	//! \brief Access originated by the direct parent to the tasks of this access sequence
	DataAccess *_superAccess;
	
	//! An identifier for the instrumentation
	Instrument::data_access_sequence_id_t _instrumentationId;
	
	
	inline DataAccessSequence();
	inline DataAccessSequence(DataAccessRange accessRange, DataAccess *superAccess = 0);
	
	
	//! \brief Reevaluate the satisfactibility of a DataAccess according to the once immediately preceeding it (if any)
	//! 
	//! \param[in] position an iterator to the list position of the DataAccess to be reevaluated
	//! 
	//! \returns true if the DataAccess has become satisfied
	inline bool reevaluateSatisfactibility(access_sequence_t::iterator position);
	
	
	//! \brief Upgrade a DataAccess to a new access type
	//! 
	//! \param[in] task the task that performs the access
	//! \param[in] position the position of the DataAccess in the sequence
	//! \param[inout] oldAccess the DataAccess to be upgraded
	//! \param[in] newAccessType the type of access that triggers the update
	//! 
	//! \returns false if the DataAccess becomes unsatisfied
	inline bool upgradeAccess(Task* task, access_sequence_t::reverse_iterator& position, DataAccess& oldAccess, DataAccessType newAccessType);
	
	
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


#include "DataAccess.hpp"
#include "DataAccessSequenceLinkingArtifactsImplementation.hpp"


#endif // DATA_ACCESS_SEQUENCE_HPP
