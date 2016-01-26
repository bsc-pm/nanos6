#ifndef DATA_ACCESS_SEQUENCE_HPP
#define DATA_ACCESS_SEQUENCE_HPP

#include <mutex>
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
	
	//! \brief Access originated by the direct parent to the tasks of this access sequence
	DataAccess *_superAccess;
	
	union {
		//! \brief SpinLock to protect the sequence and all its subaccesses. Only valid when _superAccess == nullptr
		SpinLock _rootLock;
		
		//! \brief Pointer to the root DataAccessSequence that does contain the lock that protects all of its hierarchy. Only valid when _superAccess != nullptr
		DataAccessSequence *_rootSequence;
	};
	
	typedef boost::intrusive::list<DataAccess, boost::intrusive::function_hook<DataAccessSequenceLinkingArtifacts>> access_sequence_t;
	
	//! \brief Ordered sequence of accesses to the same location
	access_sequence_t _accessSequence;
	
	//! An identifier for the instrumentation
	Instrument::data_access_sequence_id_t _instrumentationId;
	
	
	inline DataAccessSequence();
	inline DataAccessSequence(DataAccessRange accessRange);
	inline DataAccessSequence(DataAccessRange accessRange, DataAccess *superAccess);
	inline DataAccessSequence(DataAccessRange accessRange, DataAccess *superAccess, DataAccessSequence *rootSequence);
	inline ~DataAccessSequence();
	
	
	//! \brief Get the Root DataAccessSequence that leads to this one
	inline DataAccessSequence *getRootSequence();
	
	//! \brief Lock the sequence.
	//! 
	//! NOTE: The current implementation uses a single non-recursive lock for all the hierarchy of accesses to the same data
	inline void lock();
	
	//! \brief Unlok the sequence
	inline void unlock();
	
	//! \brief Get a locking guard
	inline std::unique_lock<SpinLock> getLockGuard();
	
	
	//! \brief Evaluate the satisfiability of a DataAccessType according to its effective previous (if any)
	//! 
	//! \param[in] previousDataAccess the effective previous access or nullptr if there is none
	//! \param[in] nextAccessType the type of access that will follow and whose satisfiability is to be evaluated
	//! 
	//! \returns true if the nextAccessType is satisfied
	static inline bool evaluateSatisfiability(DataAccess *previousDataAccess, DataAccessType nextAccessType);
	
	//! \brief Reevaluate the satisfiability of a DataAccess according to its effective previous (if any)
	//! 
	//! \param[in] previousDataAccess the effective previous access or nullptr if there is none
	//! \param[in] targetDataAccess the DataAccess whose satisfiability it to be reevaluated
	//! 
	//! \returns true if the DataAccess has become satisfied
	static inline bool reevaluateSatisfiability(DataAccess *previousDataAccess, DataAccess *targetDataAccess);
	
	
	//! \brief Reevaluate the satisfiability of a DataAccess according to the one immediately preceeding it (if any)
	//! 
	//! \param[in] position an iterator to the list position of the DataAccess to be reevaluated
	//! 
	//! \returns true if the DataAccess has become satisfied
	inline bool reevaluateSatisfiability(access_sequence_t::iterator position);
	
	
	inline bool upgradeSameTypeAccess(Task *task, DataAccess /* INOUT */ *dataAccess, bool newAccessWeakness);
	inline bool upgradeSameStrengthAccess(Task *task, DataAccess /* INOUT */ *dataAccess, DataAccessType newAccessType);
	inline bool upgradeStrongAccessWithWeak(Task *task, DataAccess /* INOUT */ * /* INOUT */ &dataAccess, DataAccessType newAccessType);
	inline bool upgradeWeakAccessWithStrong(Task *task, DataAccess /* INOUT */ * /* INOUT */ &dataAccess, DataAccessType newAccessType);
	
	
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
	inline bool upgradeAccess(Task* task, DataAccess /* INOUT */ * /* INOUT */ &dataAccess, DataAccessType newAccessType, bool newAccessWeakness);
	
	
	//! \brief Get the Effective Previous access of another given one
	//! 
	//! \param[in] dataAccess the DataAccess that is effectively after the one to be returned or nullptr if the DataAccess is yet to be added and the sequence is empty
	//! 
	//! \returns the Effective Previous access to the one passed by parameter, or nullptr if there is none
	//! 
	//! NOTE: This function assumes that the whole hierarchy has already been locked
	inline DataAccess *getEffectivePrevious(DataAccess *dataAccess);
	
	
};


#include "DataAccess.hpp"
#include "DataAccessSequenceLinkingArtifactsImplementation.hpp"


#endif // DATA_ACCESS_SEQUENCE_HPP
