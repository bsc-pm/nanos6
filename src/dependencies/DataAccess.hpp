#ifndef DATA_ACCESS_HPP
#define DATA_ACCESS_HPP

#ifndef DATA_ACCESS_SEQUENCE_HPP
	#warning Please, include DataAccessSequence.hpp instead of DataAccess.hpp. Otherwise it may fail to compile.
#endif // DATA_ACCESS_SEQUENCE_HPP


#include <boost/intrusive/list.hpp>
#include <boost/intrusive/list_hook.hpp>
#include <atomic>
#include <cassert>
#include <set>

#include <InstrumentDataAccessId.hpp>


struct DataAccess;
struct DataAccessSequence;
class Task;


#include "DataAccessType.hpp"
#include "DataAccessRange.hpp"


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
	
	//! \brief Countdown until full completion of the access
	//! +1 if the originator has not finished
	//! +1 if _subaccesses is not empty
	std::atomic<int> _completionCountdown;
	
	//! Tasks to which the access corresponds
	Task *_originator;
	
	//! Accesses performed by the direct children of the _originator task
	DataAccessSequence _subaccesses;
	
	//! An identifier for the instrumentation
	Instrument::data_access_id_t _instrumentationId;
	
	DataAccess(
		DataAccessSequence *dataAccessSequence,
		DataAccessType type,
		bool satisfied,
		Task *originator,
		DataAccessRange accessRange,
		Instrument::data_access_id_t instrumentationId
	)
		: _taskAccessListLinks(), _accessSequenceLinks(), _dataAccessSequence(dataAccessSequence),
		_type(type), _satisfied(satisfied), _completionCountdown(1 /* For the originator */), _originator(originator),
		_subaccesses(accessRange, this),
		_instrumentationId(instrumentationId)
	{
		assert(dataAccessSequence != 0);
		assert(originator != 0);
	}
	
};


#endif // DATA_ACCESS_HPP