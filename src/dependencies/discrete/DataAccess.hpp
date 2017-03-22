#ifndef DATA_ACCESS_HPP
#define DATA_ACCESS_HPP

#include <boost/intrusive/list.hpp>
#include <boost/intrusive/list_hook.hpp>

#include <InstrumentDataAccessId.hpp>


struct DataAccess;
struct DataAccessSequence;
class Task;


#include "../DataAccessType.hpp"
#include "../DataAccessBase.hpp"
#include "DataAccessRange.hpp"


//! The accesses that one or more tasks perform sequentially to a memory location that can occur concurrently (unless commutative).
struct DataAccess : public DataAccessBase {
	typedef boost::intrusive::list_member_hook<link_mode_t> access_sequence_links_t;
	
	//! True if the data access can already be performed
	bool _satisfied;
	
	//! Links used by the list within DataAccessSequence
	access_sequence_links_t _accessSequenceLinks;
	
	//! Pointer to the DataAccessSequence that contains this access. This is needed for locking.
	DataAccessSequence *_dataAccessSequence;
	
	//! Accesses performed by the direct children of the _originator task
	DataAccessSequence _subaccesses;
	
	//! Reduction type and operator
	int _reductionInfo;
	
	inline DataAccess(
		DataAccessSequence *dataAccessSequence,
		DataAccessType type,
		bool weak,
		bool satisfied,
		Task *originator,
		DataAccessRange accessRange,
		Instrument::data_access_id_t instrumentationId
	);
	
	inline DataAccess(
		DataAccessSequence *dataAccessSequence,
		DataAccessType type,
		bool weak,
		bool satisfied,
		Task *originator,
		DataAccessRange accessRange,
		Instrument::data_access_id_t instrumentationId,
		int reductionInfo
	);
	
	//! \brief Evaluate the satisfiability of a DataAccessType according to its effective previous (if any)
	//! 
	//! \param[in] previousDataAccess the effective previous access or nullptr if there is none
	//! \param[in] nextAccessType the type of access that will follow and whose satisfiability is to be evaluated
	//! 
	//! \returns true if the nextAccessType is satisfied
	static inline bool evaluateSatisfiability(DataAccess *effectivePrevious, DataAccessType nextAccessType);
	
	//! \brief Evaluate the satisfiability of a reduction DataAccess according to its effective previous (if any)
	//! 
	//! \param[in] previousDataAccess the effective previous access or nullptr if there is none
	//! \param[in] nextAccessType REDUCTION_ACCESS_TYPE
	//! \param[in] reductionOperation reduction information about reduction data type and operator
	//! 
	//! \returns true if the nextAccessType is satisfied
	static inline bool evaluateSatisfiability(DataAccess *effectivePrevious, DataAccessType nextAccessType, int reductionOperation);
	
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


#include "DataAccessSequence.hpp"


#endif // DATA_ACCESS_HPP
