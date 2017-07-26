/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_HPP
#define DATA_ACCESS_HPP


#include <boost/intrusive/avl_set.hpp>
#include <boost/intrusive/avl_set_hook.hpp>
#include <atomic>
#include <cassert>
#include <set>

#include <InstrumentDataAccessId.hpp>
#include <InstrumentTaskId.hpp>


struct DataAccess;
class Task;
class SpinLock;


#include "../DataAccessBase.hpp"
#include "DataAccessRange.hpp"
#include "LinearRegionDataAccessMap.hpp"


//! The accesses that one or more tasks perform sequentially to a memory location that can occur concurrently (unless commutative).
struct DataAccess : public DataAccessBase {
	//! Links equivalent to the ones within a DataAccessSequence
	DataAccessPreviousLinks _previous;
	DataAccessNextLinks _next;
	
	//! Pointer to the parent access that contains this access.
	DataAccess *_superAccess;
	
	//! Pointer to the lock that protects the hierarchy that contains this access.
	SpinLock *_lock;
	
	//! Pointer to the bottom map of accesses that allows to calculate dependencies related to this access
	LinearRegionDataAccessMap *_bottomMap;
	
	//! The range of data covered by the access
	DataAccessRange _range;
	
	//! Equal to 0 when the data access can be performed
	int _blockerCount;
	
	//! Top map of accesses performed by the direct children of the _originator task
	LinearRegionDataAccessMap _topSubaccesses;
	
	//! Bottom map of accesses performed by the direct children of the _originator task
	LinearRegionDataAccessMap _bottomSubaccesses;
	
	DataAccess(
		DataAccess *superAccess, SpinLock *lock, LinearRegionDataAccessMap *bottomMap,
		DataAccessType type, bool weak,
		int blockerCount,
		Task *originator,
		DataAccessRange accessRange,
		Instrument::data_access_id_t instrumentationId
	)
		: DataAccessBase(type, weak, originator, instrumentationId),
		_previous(), _next(),
		_superAccess(superAccess), _lock(lock), _bottomMap(bottomMap),
		_range(accessRange), _blockerCount(blockerCount),
		_topSubaccesses(this), _bottomSubaccesses(this)
	{
		assert(bottomMap != 0);
		assert(originator != 0);
	}
	
	
	DataAccessRange const &getAccessRange() const
	{
		return _range;
	}
	
	DataAccessRange &getAccessRange()
	{
		return _range;
	}
	
	
	inline void fullLinkTo(DataAccessRange const &range, DataAccess *target, bool blocker, Instrument::task_id_t triggererTaskInstrumentationId);
	
	//! \brief Pass the Effective Previous at a given range through a lambda
	//! 
	//! \param[in] range the range over which to look up the effective previous
	//! \param[in] processDirectPrevious include also the directly linked previous accesses
	//! \param[in] effectivePreviousProcessor the lambda that receives the effective previous through a DataAccessPreviousLinks::iterator and returns false if the traversal should be stoped
	//! 
	//! \returns false if the traversal was stopped before it had finished
	//!
	//! NOTE: This function assumes that the whole hierarchy has already been locked
	template <typename EffectivePreviousProcessorType>
	bool processEffectivePrevious(DataAccessRange const &range, bool processDirectPrevious, EffectivePreviousProcessorType effectivePreviousProcessor);
	
	//! \brief Updates the blocker count according to a new (upgraded) access type
	//! 
	//! \param[in] accessType the type of the new access
	//! 
	//! \returns true if the access becomes unsatisfied
	inline bool updateBlockerCount(DataAccessType accessType);
	
	//! \brief Calculates the blocker count according to a new (upgraded) access type
	//! 
	//! \param[in] accessType the type of the new access
	//! 
	//! \returns the blocker count
	inline int calculateBlockerCount(DataAccessType accessType);
	
	//! \brief Updates the blocker count and the link satisfiability of the previous accesses according to the current access type
	//! 
	//! \returns true if the access becomes unsatisfied
	inline bool updateBlockerCountAndLinkSatisfiability();
	
	//! \brief Evaluate the satisfiability of a DataAccessType according to its effective previous (if any)
	//! 
	//! \param[in] previousDataAccess the effective previous access or nullptr if there is none
	//! \param[in] nextAccessType the type of access that will follow and whose satisfiability is to be evaluated
	//! 
	//! \returns true if the nextAccessType is satisfied
	static inline bool evaluateSatisfiability(DataAccess *effectivePrevious, DataAccessType nextAccessType);
	
	static inline bool upgradeSameTypeAccess(Task *task, DataAccess /* INOUT */ *dataAccess, bool newAccessWeakness);
	static inline bool upgradeSameStrengthAccess(Task *task, DataAccess /* INOUT */ *dataAccess, DataAccessType newAccessType);
	static inline bool upgradeStrongAccessWithWeak(Task *task, DataAccess /* INOUT */ * /* INOUT */ &dataAccess, DataAccessType newAccessType);
	static inline bool upgradeWeakAccessWithStrong(Task *task, DataAccess /* INOUT */ * /* INOUT */ &dataAccess, DataAccessType newAccessType);
	
	
	//! \brief Evaluate if an access propagates its satisfiability to a following access of a given type
	//! 
	//! \param[in] nextAccessType the type of access that follows
	//! 
	//! \returns true if the an access of a given type would become satisfied by propagation when the current access becomes satisfied
	inline bool propagatesSatisfiability(DataAccessType nextAccessType);
	
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
