/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_HPP
#define DATA_ACCESS_HPP

#define MAX_SYMBOLS 64 // TODO: Temporary solution to use a fixed bitset size

#include <boost/intrusive/avl_set.hpp>
#include <boost/intrusive/avl_set_hook.hpp>

#include <atomic>
#include <bitset>
#include <cassert>
#include <set>
#include <boost/dynamic_bitset.hpp>

#include <executors/threads/CPUManager.hpp>

#include <InstrumentDataAccessId.hpp>
#include <InstrumentTaskId.hpp>

#include <lowlevel/SpinLock.hpp>

struct DataAccess;
class Task;
class ReductionInfo;


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
		COMMUTATIVE_SATISFIED_BIT,
		RECEIVED_REDUCTION_INFO_BIT,
		ALLOCATED_REDUCTION_INFO_BIT,
		RECEIVED_CPU_SET_BIT,
		CLOSES_REDUCTION_BIT,
		
		READ_SATISFIABILITY_PROPAGATION_INHIBITED_BIT,
		CONCURRENT_SATISFIABILITY_PROPAGATION_INHIBITED_BIT,
		COMMUTATIVE_SATISFIABILITY_PROPAGATION_INHIBITED_BIT,
		REDUCTION_INFO_PROPAGATION_INHIBITED_BIT,
		
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
	typedef std::bitset<MAX_SYMBOLS> symbols_t;	
	
private:
	DataAccessObjectType _objectType;
	
	//! The region of data covered by the access
	DataAccessRegion _region;
	
	status_t _status;
	
	//! Direct next access
	DataAccessLink _next;
	
	//! An index that determines the data type and the operation of the reduction (if applicable)
	reduction_type_and_operator_index_t _reductionTypeAndOperatorIndex;
	
	//! A bitmap of the "symbols" this access is related to
	symbols_t _symbols; 
	
	//! An index that identifies the reduction within the task (if applicable)
	reduction_index_t _reductionIndex;
	
	//! Reduction-specific information of current access
	ReductionInfo *_reductionInfo;
	
	//! Reduction-specific information of previous access
	ReductionInfo *_previousReductionInfo;
	
	//! CPUs executing tasks accessing this reduction region (if applicable)
	boost::dynamic_bitset<> _reductionCpuSet;
	
public:
	DataAccess(
		DataAccessObjectType objectType,
		DataAccessType type, bool weak,
		Task *originator,
		DataAccessRegion accessRegion,
		reduction_type_and_operator_index_t reductionTypeAndOperatorIndex,
		reduction_index_t reductionIndex,
		Instrument::data_access_id_t instrumentationId = Instrument::data_access_id_t(),
		status_t status = 0, DataAccessLink next = DataAccessLink()
	)
		: DataAccessBase(type, weak, originator, instrumentationId),
		_objectType(objectType),
		_region(accessRegion),
		_status(status),
		_next(next),
		_reductionTypeAndOperatorIndex(reductionTypeAndOperatorIndex),
		_reductionIndex(reductionIndex),
		_reductionInfo(nullptr),
		_previousReductionInfo(nullptr)
	{
		assert(originator != nullptr);
		
		if (_type == REDUCTION_ACCESS_TYPE) {
			_reductionCpuSet.resize(CPUManager::getTotalCPUs());
		}
	}
	
	DataAccess(const DataAccess &other)
		: DataAccessBase(other.getType(), other.isWeak(), other.getOriginator(), Instrument::data_access_id_t()),
		_objectType(other.getObjectType()),
		_region(other.getAccessRegion()),
		_status(other.getStatus()),
		_next(other.getNext()),
		_reductionTypeAndOperatorIndex(other.getReductionTypeAndOperatorIndex()),
		_reductionIndex(other.getReductionIndex()),
		_reductionInfo(other.getReductionInfo()),
		_previousReductionInfo(other.getPreviousReductionInfo()),
		_reductionCpuSet(other.getReductionCpuSet())
	{}
	
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
			nullptr,
			_type, _weak, _region,
			/* Read Satisfied */ false, /* Write Satisfied */ false, /* Globally Satisfied */ false,
			(Instrument::access_object_type_t) _objectType,
			taskInstrumentationId
		);
	}
	
	inline void setUpNewFragment(Instrument::data_access_id_t originalAccessInstrumentationId)
	{
		_instrumentationId = Instrument::fragmentedDataAccess(originalAccessInstrumentationId, _region);
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
		Instrument::newDataAccessProperty(_instrumentationId, "RSat", "Read Satisfied");
	}
	bool readSatisfied() const
	{
		return _status[READ_SATISFIED_BIT];
	}
	
	void setWriteSatisfied()
	{
		assert(!writeSatisfied());
		_status[WRITE_SATISFIED_BIT] = true;
		Instrument::newDataAccessProperty(_instrumentationId, "WSat", "Write Satisfied");
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
	
	void setCommutativeSatisfied()
	{
		assert(!commutativeSatisfied());
		_status[COMMUTATIVE_SATISFIED_BIT] = true;
		Instrument::newDataAccessProperty(_instrumentationId, "CommSat", "Commutative Satisfied");
	}
	bool commutativeSatisfied() const
	{
		return _status[COMMUTATIVE_SATISFIED_BIT];
	}
	
	void setReceivedReductionInfo()
	{
		assert(!receivedReductionInfo());
		_status[RECEIVED_REDUCTION_INFO_BIT] = true;
		Instrument::newDataAccessProperty(_instrumentationId, "RIRec", "ReductionInfo Received");
	}
	bool receivedReductionInfo() const
	{
		return _status[RECEIVED_REDUCTION_INFO_BIT];
	}
	
	void setAllocatedReductionInfo()
	{
		assert(_type == REDUCTION_ACCESS_TYPE);
		assert(!allocatedReductionInfo());
		_status[ALLOCATED_REDUCTION_INFO_BIT] = true;
		Instrument::newDataAccessProperty(_instrumentationId, "RIAlloc", "ReductionInfo Allocated");
	}
	bool allocatedReductionInfo() const
	{
		return _status[ALLOCATED_REDUCTION_INFO_BIT];
	}
	
	void setReceivedReductionCpuSet()
	{
		assert(!receivedReductionCpuSet());
		_status[RECEIVED_CPU_SET_BIT] = true;
		Instrument::newDataAccessProperty(_instrumentationId, "RSetRec", "Reduction Set Received");
	}
	bool receivedReductionCpuSet() const
	{
		return _status[RECEIVED_CPU_SET_BIT];
	}
	
	void setClosesReduction()
	{
		assert(_type == REDUCTION_ACCESS_TYPE);
		assert(!closesReduction());
		_status[CLOSES_REDUCTION_BIT] = true;
		Instrument::newDataAccessProperty(_instrumentationId, "Rc", "Closes Reduction");
	}
	bool closesReduction() const
	{
		return _status[CLOSES_REDUCTION_BIT];
	}
	
	bool canPropagateReadSatisfiability() const
	{
		return !_status[READ_SATISFIABILITY_PROPAGATION_INHIBITED_BIT];
	}
	void unsetCanPropagateReadSatisfiability()
	{
		assert(canPropagateReadSatisfiability());
		_status[READ_SATISFIABILITY_PROPAGATION_INHIBITED_BIT] = true;
	}
	
	bool canPropagateConcurrentSatisfiability() const
	{
		return !_status[CONCURRENT_SATISFIABILITY_PROPAGATION_INHIBITED_BIT];
	}
	void unsetCanPropagateConcurrentSatisfiability()
	{
		assert(canPropagateConcurrentSatisfiability());
		_status[CONCURRENT_SATISFIABILITY_PROPAGATION_INHIBITED_BIT] = true;
	}
	
	bool canPropagateCommutativeSatisfiability() const
	{
		return !_status[COMMUTATIVE_SATISFIABILITY_PROPAGATION_INHIBITED_BIT];
	}
	void unsetCanPropagateCommutativeSatisfiability()
	{
		assert(canPropagateCommutativeSatisfiability());
		_status[COMMUTATIVE_SATISFIABILITY_PROPAGATION_INHIBITED_BIT] = true;
	}
	
	bool canPropagateReductionInfo() const
	{
		return !_status[REDUCTION_INFO_PROPAGATION_INHIBITED_BIT];
	}
	void unsetCanPropagateReductionInfo()
	{
		assert(canPropagateReductionInfo());
		_status[REDUCTION_INFO_PROPAGATION_INHIBITED_BIT] = true;
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
		if (other->commutativeSatisfied()) {
			setCommutativeSatisfied();
		}
		if (other->receivedReductionInfo()) {
			setReceivedReductionInfo();
			setPreviousReductionInfo(other->getPreviousReductionInfo());
		}
		if (other->getReductionInfo() != nullptr) {
			setReductionInfo(other->getReductionInfo());
		}
		if (other->allocatedReductionInfo()) {
			assert(other->getReductionInfo() != nullptr);
			setAllocatedReductionInfo();
		}
		if (other->receivedReductionCpuSet()) {
			setReceivedReductionCpuSet();
		}
		if (other->getReductionCpuSet().size() > 0) {
			setReductionCpuSet(other->getReductionCpuSet());
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
		} else if (_type == COMMUTATIVE_ACCESS_TYPE) {
			return commutativeSatisfied();
		} else if (_type == REDUCTION_ACCESS_TYPE) {
			// Semantic note: For a reduction access to be satisfied it means
			// that the necessary structures for executing and combining the
			// reduction have been received and that the original region can be
			// safely accessed for combination
			return writeSatisfied()
				&& (allocatedReductionInfo() || (receivedReductionInfo() && receivedReductionCpuSet()));
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
	
	reduction_index_t getReductionIndex() const
	{
		return _reductionIndex;
	}
	
	ReductionInfo *getReductionInfo() const
	{
		return _reductionInfo;
	}
	
	void setReductionInfo(ReductionInfo *reductionInfo)
	{
		assert(_reductionInfo == nullptr);
		assert(_type == REDUCTION_ACCESS_TYPE);
		_reductionInfo = reductionInfo;
	}
	
	ReductionInfo *getPreviousReductionInfo() const
	{
		return _previousReductionInfo;
	}
	
	void setPreviousReductionInfo(ReductionInfo *previousReductionInfo)
	{
		assert(_previousReductionInfo == nullptr);
		_previousReductionInfo = previousReductionInfo;
	}
	
	boost::dynamic_bitset<> const &getReductionCpuSet() const
	{
		return _reductionCpuSet;
	}
	
	boost::dynamic_bitset<> &getReductionCpuSet()
	{
		return _reductionCpuSet;
	}
	
	void setReductionCpuSet(const boost::dynamic_bitset<> &reductionCpuSet)
	{
		assert(_reductionCpuSet.none());
		_reductionCpuSet = reductionCpuSet;
	}
	
	void setReductionCpu(size_t cpuId)
	{
		_reductionCpuSet.set(cpuId);
	}
	
	Instrument::data_access_id_t const &getInstrumentationId() const
	{
		return _instrumentationId;
	}
	
	Instrument::data_access_id_t &getInstrumentationId()
	{
		return _instrumentationId;
	}
	
	bool isInSymbol(int symbol) const 
	{
		return _symbols[symbol];
	}
	void addToSymbol(int symbol)
	{
		_symbols.set(symbol);
	}
	void removeFromSymbol(int symbol)
	{
		_symbols.reset(symbol);
	}
	void addToSymbols(const symbols_t &symbols)
	{
		_symbols |= symbols;
	}
	symbols_t getSymbols() const
	{
		return _symbols;
	}	

};


#endif // DATA_ACCESS_HPP
