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
#include "DataAccessLink.hpp"
#include "DataAccessObjectType.hpp"
#include "DataAccessRegion.hpp"
#include "ReductionSpecific.hpp"

#include <InstrumentDependenciesByAccessLinks.hpp>


//! The accesses that one or more tasks perform sequentially to a memory location that can occur concurrently (unless commutative).
struct DataAccess : protected DataAccessBase {
	friend struct TaskDataAccessLinkingArtifacts;
	
private:
	enum status_bit_coding {
		REGISTERED_BIT = 0,
		
		COMPLETE_BIT,
		
		READ_SATISFIED_BIT,
		WRITE_SATISFIED_BIT,
		CONCURRENT_SATISFIED_BIT,
		ANY_REDUCTION_SATISFIED_BIT,
		MATCHING_REDUCTION_SATISFIED_BIT,
		
		READ_PROPAGATION_INHIBITED_BIT,
		CONCURRENT_PROPAGATION_INHIBITED_BIT,
		ANY_REDUCTION_PROPAGATION_INHIBITED_BIT,
		MATCHING_REDUCTION_PROPAGATION_INHIBITED_BIT,
		
		HAS_SUBACCESSES_BIT,
		IN_BOTTOM_MAP_BIT,
		TOPMOST_BIT,
		TOP_LEVEL_BIT,
#ifndef NDEBUG
		IS_REACHABLE_BIT,
		HAS_BEEN_DISCOUNTED_BIT,
#endif
		TOTAL_STATUS_BITS
	};
	
public:
	typedef std::bitset<TOTAL_STATUS_BITS> status_t;
	
	
private:
	DataAccessObjectType _objectType;
	
	//! The region of data covered by the access
	DataAccessRegion _region;
	
	status_t _status;
	
	//! Direct next access
	DataAccessLink _next;
	
	//! An index that determines the data type and the operation of the reduction (if applicable)
	reduction_type_and_operator_index_t _reductionTypeAndOperatorIndex;
	
	
public:
	DataAccess(
		DataAccessObjectType objectType,
		DataAccessType type, bool weak,
		Task *originator,
		DataAccessRegion accessRegion,
		reduction_type_and_operator_index_t reductionTypeAndOperatorIndex,
		Instrument::data_access_id_t instrumentationId = Instrument::data_access_id_t(),
		status_t status = 0, DataAccessLink next = DataAccessLink()
	)
		: DataAccessBase(type, weak, originator, instrumentationId),
		_objectType(objectType),
		_region(accessRegion),
		_status(status),
		_next(next),
		_reductionTypeAndOperatorIndex(reductionTypeAndOperatorIndex)
	{
		assert(originator != nullptr);
	}
	
	~DataAccess()
	{
		Instrument::removedDataAccess(_instrumentationId);
		assert(hasBeenDiscounted());
	}
	
	inline DataAccessObjectType getObjectType() const
	{
		return _objectType;
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
			_type, _weak, _region,
			/* Read Satisfied */ false, /* Write Satisfied */ false, /* Globally Satisfied */ false,
			(Instrument::access_object_type_t) _objectType,
			taskInstrumentationId
		);
	}
	
	inline void setUpNewFragment(Instrument::data_access_id_t originalAccessInstrumentationId)
	{
		_instrumentationId = Instrument::fragmentedDataAccess(originalAccessInstrumentationId, _region);
		if (isTopmost()) {
			Instrument::newDataAccessProperty(_instrumentationId, "T", "Topmost");
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
	
	void setRegistered()
	{
		assert(!isRegistered());
		_status[REGISTERED_BIT] = true;
	}
	bool isRegistered() const
	{
		return _status[REGISTERED_BIT];
	}
	void clearRegistered()
	{
		// No assertion here since it is a clear method instead of an unset method
		_status[REGISTERED_BIT] = false;
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
	
	bool canPropagateReadSatisfiability() const
	{
		return !_status[READ_PROPAGATION_INHIBITED_BIT];
	}
	void unsetCanPropagateReadSatisfiability()
	{
		assert(canPropagateReadSatisfiability());
		_status[READ_PROPAGATION_INHIBITED_BIT] = true;
	}
	
	bool canPropagateConcurrentSatisfiability() const
	{
		return !_status[CONCURRENT_PROPAGATION_INHIBITED_BIT];
	}
	void unsetCanPropagateConcurrentSatisfiability()
	{
		assert(canPropagateConcurrentSatisfiability());
		_status[CONCURRENT_PROPAGATION_INHIBITED_BIT] = true;
	}
	
	bool canPropagateAnyReductionSatisfiability() const
	{
		return !_status[ANY_REDUCTION_PROPAGATION_INHIBITED_BIT];
	}
	void unsetCanPropagateAnyReductionSatisfiability()
	{
		assert(canPropagateAnyReductionSatisfiability());
		_status[ANY_REDUCTION_PROPAGATION_INHIBITED_BIT] = true;
	}
	
	bool canPropagateMatchingReductionSatisfiability() const
	{
		return !_status[MATCHING_REDUCTION_PROPAGATION_INHIBITED_BIT];
	}
	void unsetCanPropagateMatchingReductionSatisfiability()
	{
		assert(canPropagateMatchingReductionSatisfiability());
		_status[MATCHING_REDUCTION_PROPAGATION_INHIBITED_BIT] = true;
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
		_status[IN_BOTTOM_MAP_BIT] = true;
	}
	void unsetInBottomMap()
	{
		assert(isInBottomMap());
		_status[IN_BOTTOM_MAP_BIT] = false;
	}
	bool isInBottomMap() const
	{
		return _status[IN_BOTTOM_MAP_BIT];
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
	
	void setTopLevel()
	{
		assert(!isTopLevel());
		_status[TOP_LEVEL_BIT] = true;
	}
	void clearTopLevel()
	{
		_status[TOP_LEVEL_BIT] = false;
	}
	bool isTopLevel() const
	{
		return _status[TOP_LEVEL_BIT];
	}
	
	
#ifndef NDEBUG
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
	
	DataAccessRegion const &getAccessRegion() const
	{
		return _region;
	}
	
	void setAccessRegion(DataAccessRegion const &newRegion)
	{
		_region = newRegion;
		if (_instrumentationId != Instrument::data_access_id_t()) {
			Instrument::modifiedDataAccessRegion(_instrumentationId, _region);
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
	
	
	bool hasNext() const
	{
		return (_next._task != nullptr);
	}
	void setNext(DataAccessLink const &next)
	{
		_next = next;
	}
	DataAccessLink const &getNext() const
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
	
};


#endif // DATA_ACCESS_HPP
