/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_HPP
#define DATA_ACCESS_HPP


#include <boost/intrusive/avl_set.hpp>
#include <boost/intrusive/avl_set_hook.hpp>

#include <atomic>
#include <bitset>
#include <cassert>
#include <set>

#include <InstrumentDataAccessId.hpp>
#include <InstrumentTaskId.hpp>

#include <lowlevel/SpinLock.hpp>

struct DataAccess;
class Task;


#include "../DataAccessBase.hpp"
#include "DataAccessRange.hpp"
#include "ReductionSpecific.hpp"

#include <InstrumentDependenciesByAccessLinks.hpp>


//! The accesses that one or more tasks perform sequentially to a memory location that can occur concurrently (unless commutative).
struct DataAccess : protected DataAccessBase {
	friend struct TaskDataAccessLinkingArtifacts;
	
private:
	enum status_bit_coding {
		COMPLETE_BIT = 0,
		READ_SATISFIED_BIT,
		WRITE_SATISFIED_BIT,
		CONCURRENT_SATISFIED_BIT,
		ANY_REDUCTION_SATISFIED_BIT,
		MATCHING_REDUCTION_SATISFIED_BIT,
		FRAGMENT_BIT,
		HAS_SUBACCESSES_BIT,
		IN_BOTTOM_MAP,
		TOPMOST_BIT,
		FORCE_REMOVAL_BIT,
		HAS_PROPAGATED_READ_SATISFIABILITY_BIT,
		HAS_PROPAGATED_WRITE_SATISFIABILITY_BIT,
		HAS_PROPAGATED_CONCURRENT_SATISFIABILITY_BIT,
		HAS_PROPAGATED_ANY_REDUCTION_SATISFIABILITY_BIT,
		HAS_PROPAGATED_MATCHING_REDUCTION_SATISFIABILITY_BIT,
#ifndef NDEBUG
		HAS_PROPAGATED_TOPMOST_PROPERTY_BIT,
		IS_REACHABLE_BIT,
		HAS_BEEN_DISCOUNTED_BIT,
#endif
		TOTAL_STATUS_BITS
	};
	
public:
	typedef std::bitset<TOTAL_STATUS_BITS> status_t;
	
	
private:
	//! The range of data covered by the access
	DataAccessRange _range;
	
	status_t _status;
	
	//! Direct next access
	Task *_next;
	
	//! An index that determines the data type and the operation of the reduction (if applicable)
	reduction_type_and_operator_index_t _reductionTypeAndOperatorIndex;
	
	
public:
	DataAccess(
		DataAccessType type, bool weak,
		Task *originator,
		DataAccessRange accessRange,
		bool fragment,
		reduction_type_and_operator_index_t reductionTypeAndOperatorIndex,
		Instrument::data_access_id_t instrumentationId = Instrument::data_access_id_t(),
		status_t status = 0, Task *next = nullptr
	)
		: DataAccessBase(type, weak, originator, instrumentationId),
		_range(accessRange),
		_status(status),
		_next(next),
		_reductionTypeAndOperatorIndex(reductionTypeAndOperatorIndex)
	{
		assert(originator != nullptr);
		
		if (fragment) {
			_status[FRAGMENT_BIT] = true;
		}
	}
	
	~DataAccess()
	{
		Instrument::removedDataAccess(_instrumentationId);
	}
	
	inline DataAccessType getType() const
	{
		return _type;
	}
	
	inline bool isWeak() const
	{
		return _weak;
	}
	
	inline Task *getOriginator() const
	{
		return _originator;
	}
	
	inline void setNewInstrumentationId(Instrument::task_id_t const &taskInstrumentationId)
	{
		_instrumentationId = Instrument::createdDataAccess(
			Instrument::data_access_id_t(),
			_type, _weak, _range,
			false, false, /* false, */ false,
			taskInstrumentationId
		);
	}
	
	inline void setUpNewFragment(Instrument::data_access_id_t originalAccessInstrumentationId)
	{
		_instrumentationId = Instrument::fragmentedDataAccess(originalAccessInstrumentationId, _range);
		if (isTopmost()) {
			Instrument::newDataAccessProperty(_instrumentationId, "T", "Topmost");
		}
		if (hasPropagatedReadSatisfiability()) {
			Instrument::newDataAccessProperty(_instrumentationId, "PropR", "Propagated Read Satisfiability");
		}
		if (hasPropagatedWriteSatisfiability()) {
			Instrument::newDataAccessProperty(_instrumentationId, "PropW", "Propagated Write Satisfiability");
		}
	}
	
	inline bool upgrade(bool newWeak, DataAccessType newType)
	{
		if ((newWeak != _weak) || (newType != _type)) {
			bool oldWeak = _weak;
			DataAccessType oldType = _type;
			
			bool wasSatisfied = satisfied();
			
			_weak = newWeak;
			_type = newType;
			
			Instrument::upgradedDataAccess(
				_instrumentationId,
				oldType, oldWeak,
				newType, newWeak,
				wasSatisfied && !satisfied()
			);
			
			return true;
		}
		
		return false;
	}
	
	status_t const &getStatus() const
	{
		return _status;
	}
	
	void setComplete()
	{
		assert(!complete());
		_status[COMPLETE_BIT] = true;
		Instrument::completedDataAccess(_instrumentationId);
	}
	bool complete() const
	{
		return _status[COMPLETE_BIT];
	}
	
	void setReadSatisfied()
	{
		assert(!readSatisfied());
		_status[READ_SATISFIED_BIT] = true;
	}
	bool readSatisfied() const
	{
		return _status[READ_SATISFIED_BIT];
	}
	
	void setWriteSatisfied()
	{
		assert(!writeSatisfied());
		_status[WRITE_SATISFIED_BIT] = true;
	}
	bool writeSatisfied() const
	{
		return _status[WRITE_SATISFIED_BIT];
	}
	
	void setConcurrentSatisfied()
	{
		assert(!concurrentSatisfied());
		_status[CONCURRENT_SATISFIED_BIT] = true;
		Instrument::newDataAccessProperty(_instrumentationId, "ConSat", "Concurrent Satisfied");
	}
	bool concurrentSatisfied() const
	{
		return _status[CONCURRENT_SATISFIED_BIT];
	}
	
	void setAnyReductionSatisfied()
	{
		assert(!anyReductionSatisfied());
		_status[ANY_REDUCTION_SATISFIED_BIT] = true;
		Instrument::newDataAccessProperty(_instrumentationId, "ARSat", "Any Reduction Satisfied");
	}
	bool anyReductionSatisfied() const
	{
		return _status[ANY_REDUCTION_SATISFIED_BIT];
	}
	
	void setMatchingReductionSatisfied()
	{
		assert(!matchingReductionSatisfied());
		_status[MATCHING_REDUCTION_SATISFIED_BIT] = true;
		Instrument::newDataAccessProperty(_instrumentationId, "MRSat", "Matching Reduction Satisfied");
	}
	bool matchingReductionSatisfied() const
	{
		return _status[MATCHING_REDUCTION_SATISFIED_BIT];
	}
	
	bool isFragment() const
	{
		return _status[FRAGMENT_BIT];
	}
	
	void setHasSubaccesses()
	{
		assert(!hasSubaccesses());
		_status[HAS_SUBACCESSES_BIT] = true;
	}
	void unsetHasSubaccesses()
	{
		assert(hasSubaccesses());
		_status[HAS_SUBACCESSES_BIT] = false;
	}
	bool hasSubaccesses() const
	{
		return _status[HAS_SUBACCESSES_BIT];
	}
	
	void setInBottomMap()
	{
		assert(!isInBottomMap());
		_status[IN_BOTTOM_MAP] = true;
	}
	void unsetInBottomMap()
	{
		assert(isInBottomMap());
		_status[IN_BOTTOM_MAP] = false;
	}
	bool isInBottomMap() const
	{
		return _status[IN_BOTTOM_MAP];
	}
	
	
	void setTopmost()
	{
		assert(!isTopmost());
		_status[TOPMOST_BIT] = true;
		Instrument::newDataAccessProperty(_instrumentationId, "T", "Topmost");
	}
	bool isTopmost() const
	{
		return _status[TOPMOST_BIT];
	}
	
	void forceRemoval()
	{
		assert(!hasForcedRemoval());
		_status[FORCE_REMOVAL_BIT] = true;
		Instrument::newDataAccessProperty(_instrumentationId, "F", "Forced Removal");
	}
	bool hasForcedRemoval() const
	{
		return _status[FORCE_REMOVAL_BIT];
	}
	
	void setPropagatedReadSatisfiability()
	{
		assert(!hasPropagatedReadSatisfiability());
		_status[HAS_PROPAGATED_READ_SATISFIABILITY_BIT] = true;
		Instrument::newDataAccessProperty(_instrumentationId, "PropR", "Propagated Read Satisfiability");
	}
	bool hasPropagatedReadSatisfiability() const
	{
		return _status[HAS_PROPAGATED_READ_SATISFIABILITY_BIT];
	}
	
	void setPropagatedWriteSatisfiability()
	{
		assert(!hasPropagatedWriteSatisfiability());
		_status[HAS_PROPAGATED_WRITE_SATISFIABILITY_BIT] = true;
		Instrument::newDataAccessProperty(_instrumentationId, "PropW", "Propagated Write Satisfiability");
	}
	bool hasPropagatedWriteSatisfiability() const
	{
		return _status[HAS_PROPAGATED_WRITE_SATISFIABILITY_BIT];
	}
	
	void setPropagatedConcurrentSatisfiability()
	{
		assert(!hasPropagatedConcurrentSatisfiability());
		_status[HAS_PROPAGATED_CONCURRENT_SATISFIABILITY_BIT] = true;
		Instrument::newDataAccessProperty(_instrumentationId, "PropC", "Propagated Concurrent Satisfiability");
	}
	bool hasPropagatedConcurrentSatisfiability() const
	{
		return _status[HAS_PROPAGATED_CONCURRENT_SATISFIABILITY_BIT];
	}
	
	void setPropagatedAnyReductionSatisfiability()
	{
		assert(!hasPropagatedAnyReductionSatisfiability());
		_status[HAS_PROPAGATED_ANY_REDUCTION_SATISFIABILITY_BIT] = true;
		Instrument::newDataAccessProperty(_instrumentationId, "PropAR", "Propagated Any Reduction Satisfiability");
	}
	bool hasPropagatedAnyReductionSatisfiability() const
	{
		return _status[HAS_PROPAGATED_ANY_REDUCTION_SATISFIABILITY_BIT];
	}
	
	void setPropagatedMatchingReductionSatisfiability()
	{
		assert(!hasPropagatedMatchingReductionSatisfiability());
		_status[HAS_PROPAGATED_MATCHING_REDUCTION_SATISFIABILITY_BIT] = true;
		Instrument::newDataAccessProperty(_instrumentationId, "PropMR", "Propagated Matching Reduction Satisfiability");
	}
	bool hasPropagatedMatchingReductionSatisfiability() const
	{
		return _status[HAS_PROPAGATED_MATCHING_REDUCTION_SATISFIABILITY_BIT];
	}
	
	
#ifndef NDEBUG
	void setPropagatedTopmostProperty()
	{
		assert(!hasPropagatedTopmostProperty());
		_status[HAS_PROPAGATED_TOPMOST_PROPERTY_BIT] = true;
		Instrument::newDataAccessProperty(_instrumentationId, "PropT", "Propagated Topmost Property");
	}
	bool hasPropagatedTopmostProperty() const
	{
		return _status[HAS_PROPAGATED_TOPMOST_PROPERTY_BIT];
	}
	
	void setReachable()
	{
		assert(!isReachable());
		_status[IS_REACHABLE_BIT] = true;
	}
	bool isReachable() const
	{
		return _status[IS_REACHABLE_BIT];
	}
#endif
	
	void markAsDiscounted()
	{
#ifndef NDEBUG
		assert(!_status[HAS_BEEN_DISCOUNTED_BIT]);
		_status[HAS_BEEN_DISCOUNTED_BIT] = true;
#endif
		Instrument::dataAccessBecomesRemovable(_instrumentationId);
	}
	
#ifndef NDEBUG
	bool hasBeenDiscounted() const
	{
		return _status[HAS_BEEN_DISCOUNTED_BIT];
	}
#endif
	
	void inheritFragmentStatus(DataAccess const *other)
	{
		if (other->readSatisfied()) {
			setReadSatisfied();
		}
		if (other->writeSatisfied()) {
			setWriteSatisfied();
		}
		if (other->concurrentSatisfied()) {
			setConcurrentSatisfied();
		}
		if (other->anyReductionSatisfied()) {
			setAnyReductionSatisfied();
		}
		if (other->matchingReductionSatisfied()) {
			setMatchingReductionSatisfied();
		}
		if (other->complete()) {
			setComplete();
		}
	}
	
	DataAccessRange const &getAccessRange() const
	{
		return _range;
	}
	
	void setAccessRange(DataAccessRange const &newRange)
	{
		_range = newRange;
		if (_instrumentationId != Instrument::data_access_id_t()) {
			Instrument::modifiedDataAccessRange(_instrumentationId, _range);
		}
	}
	
	
	bool satisfied() const
	{
		if (_type == READ_ACCESS_TYPE) {
			return readSatisfied();
		} else if (_type == CONCURRENT_ACCESS_TYPE) {
			return concurrentSatisfied();
		} else if (_type == REDUCTION_ACCESS_TYPE) {
			return (anyReductionSatisfied() || matchingReductionSatisfied());
		} else {
			return readSatisfied() && writeSatisfied();
		}
	}
	
	
	bool hasAlreadyPropagated(
		bool assumeHasPropagatedReadSatisfiability = false,
		bool assumeHasPropagatedWriteSatisfiability = false
	) const {
		return
			(assumeHasPropagatedReadSatisfiability || hasPropagatedReadSatisfiability())
			&& (assumeHasPropagatedWriteSatisfiability || hasPropagatedWriteSatisfiability());
	}
	
	
	bool hasNext() const
	{
		return (_next != nullptr);
	}
	void setNext(Task *next)
	{
		_next = next;
	}
	Task *getNext() const
	{
		return _next;
	}
	
	reduction_type_and_operator_index_t getReductionTypeAndOperatorIndex() const
	{
		return _reductionTypeAndOperatorIndex;
	}
	
	Instrument::data_access_id_t const &getInstrumentationId() const
	{
		return _instrumentationId;
	}
	
	bool isRemovable(
		bool forceRemoval,
		bool assumeHasPropagatedReadSatisfiability = false,
		bool assumeHasPropagatedWriteSatisfiability = false
	) const {
		return isTopmost() 
			&& readSatisfied() && writeSatisfied()
			&& complete()
			&& (
					forceRemoval
					||
					( (!isInBottomMap() || hasNext()) && hasAlreadyPropagated(assumeHasPropagatedReadSatisfiability, assumeHasPropagatedWriteSatisfiability))
				)
		;
	}
	
};


#endif // DATA_ACCESS_HPP
