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
	
	//! True iff the access is weak
	bool _weak;
	
	//! If the data access can already be performed
	bool _satisfied;
	
	//! Tasks to which the access corresponds
	Task *_originator;
	
	//! Accesses performed by the direct children of the _originator task
	DataAccessSequence _subaccesses;
	
	//! An identifier for the instrumentation
	Instrument::data_access_id_t _instrumentationId;
	
	DataAccess(
		DataAccessSequence *dataAccessSequence,
		DataAccessType type,
		bool weak,
		bool satisfied,
		Task *originator,
		DataAccessRange accessRange,
		Instrument::data_access_id_t instrumentationId
	)
		: _taskAccessListLinks(), _accessSequenceLinks(), _dataAccessSequence(dataAccessSequence),
		_type(type), _weak(weak), _satisfied(satisfied), _originator(originator),
		_subaccesses(accessRange, this, dataAccessSequence->_lock),
		_instrumentationId(instrumentationId)
	{
		assert(dataAccessSequence != 0);
		assert(originator != 0);
	}
	
	
	//! \brief Evaluate the satisfiability of a DataAccessType according to its effective previous (if any)
	//! 
	//! \param[in] previousDataAccess the effective previous access or nullptr if there is none
	//! \param[in] nextAccessType the type of access that will follow and whose satisfiability is to be evaluated
	//! 
	//! \returns true if the nextAccessType is satisfied
	static inline bool evaluateSatisfiability(DataAccess *effectivePrevious, DataAccessType nextAccessType);
	
	//! \brief Reevaluate the satisfiability of a DataAccess according to its effective previous (if any)
	//! 
	//! \param[in] effectivePrevious the effective previous access or nullptr if there is none
	//! 
	//! \returns true if the DataAccess has become satisfied
	inline bool reevaluateSatisfiability(DataAccess *effectivePrevious);
	
	
	static inline bool upgradeSameTypeAccess(Task *task, DataAccess /* INOUT */ *dataAccess, bool newAccessWeakness);
	static inline bool upgradeSameStrengthAccess(Task *task, DataAccess /* INOUT */ *dataAccess, DataAccessType newAccessType);
	static inline bool upgradeStrongAccessWithWeak(Task *task, DataAccess /* INOUT */ * /* INOUT */ &dataAccess, DataAccessType newAccessType);
	static inline bool upgradeWeakAccessWithStrong(Task *task, DataAccess /* INOUT */ * /* INOUT */ &dataAccess, DataAccessType newAccessType);
	
	
	//! \brief Upgrade a DataAccess to a new access type
	//! 
	//! \param[in] task the task that performs the access
	//! \param[inout] dataAccess the DataAccess to be upgraded
	//! \param[in] newAccessType the type of access that triggers the update
	//! \param[in] newAccessWeakness true iff the access that triggers the update is weak
	//! 
	//! \returns false if the DataAccess becomes unsatisfied
	//! 
	//! NOTE: In some cases, the upgrade can create an additional DataAccess. In that case, dataAccess is updated to point to the new object.
	static inline bool upgradeAccess(Task* task, DataAccess /* INOUT */ * /* INOUT */ &dataAccess, DataAccessType newAccessType, bool newAccessWeakness);
	
};


#endif // DATA_ACCESS_HPP
