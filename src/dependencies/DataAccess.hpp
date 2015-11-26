#ifndef DATA_ACCESS_HPP
#define DATA_ACCESS_HPP

#include <boost/intrusive/list.hpp>
#include <boost/intrusive/list_hook.hpp>
#include <cassert>
#include <set>

#include <InstrumentDataAccessId.hpp>


struct DataAccess;
struct DataAccessSequence;
class Task;


#include "DataAccessType.hpp"


struct DataAccessAndParticipationPosition {
	//! An access that a task performs that can produce dependencies
	DataAccess *_dataAccess;
	
	//! The position of the task in the list of participants of the access, to speed up removal.
	std::set<Task *>::iterator _participationPosition;
};


//! The accesses that one or more tasks perform sequentially to a memory location that can occur concurrently (unless commutative).
struct DataAccess {
	#if NDEBUG
		typedef boost::intrusive::link_mode<boost::intrusive::normal_link> link_mode_t;
	#else
		typedef boost::intrusive::link_mode<boost::intrusive::safe_link> link_mode_t;
	#endif
	
	typedef boost::intrusive::list_member_hook<link_mode_t> task_access_list_links_t;
	typedef boost::intrusive::list_member_hook<link_mode_t> access_sequence_links_t;
	
	
	//! Links used by the list of accesses of a Task
	task_access_list_links_t _taskAccessListLinks;
	
	//! Links used by the list within DataAccessSequence
	access_sequence_links_t _accessSequenceLinks;
	
	//! Pointer to the DataAccessSequence that contains this access. This is needed for locking.
	DataAccessSequence *_dataAccessSequence;
	
	//! Type of access: read, write, ...
	DataAccessType _type;
	
	//! If the data access can already be performed
	bool _satisfied;
	
	//! Tasks to which the access corresponds
	Task *_originator;
	
	//! An identifier for the instrumentation
	Instrument::data_access_id_t _instrumentationId;
	
	DataAccess(
		DataAccessSequence *dataAccessSequence,
		DataAccessType type,
		bool satisfied,
		Task *originator,
		Instrument::data_access_id_t instrumentationId
	)
		: _taskAccessListLinks(), _accessSequenceLinks(), _dataAccessSequence(dataAccessSequence),
		_type(type), _satisfied(satisfied), _originator(originator),
		_instrumentationId(instrumentationId)
	{
		assert(dataAccessSequence != 0);
		assert(originator != 0);
	}
	
};


#endif // DATA_ACCESS_HPP
