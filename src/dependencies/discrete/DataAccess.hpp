/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_HPP
#define DATA_ACCESS_HPP

#include <atomic>
#include <bitset>
#include <cassert>
#include <iostream>

#include "../DataAccessType.hpp"
#include "ReductionSpecific.hpp"
#include "ReductionInfo.hpp"
#include <InstrumentDataAccessId.hpp>

struct DataAccess;
class Task;

//! Definitions for the access flags
#define BIT(n) ((unsigned int) (1 << n))

#define ACCESS_NONE					((unsigned int) 0)
#define ACCESS_READ_SATISFIED 		BIT(0) // Read satisfiability
#define ACCESS_WRITE_SATISFIED 		BIT(1) // Write satisfiability
#define ACCESS_DELETABLE	 		BIT(2) // Indicates the access is not needed in the chain
#define ACCESS_UNREGISTERED	 		BIT(3) // Indicates flags have been read unregistering
#define ACCESS_UNREGISTERING_DONE	BIT(4) // Prevents overtaking when racing in the unregister
#define ACCESS_CHILDS_FINISHED		BIT(5) // Indicates all the childs of the access have finished
#define ACCESS_IN_TASKWAIT			BIT(6) // Indicates access has blocked in a taskwait
#define ACCESS_HOLDOFF				BIT(7) // Prevents racing to delete an access

typedef unsigned int access_flags_t;

//! The accesses that one or more tasks perform sequentially to a memory location that can occur concurrently (unless commutative).
struct DataAccess {
private:
	//! The type of the access
	DataAccessType _type;

	//! The originator of the access
	Task *_originator;

	//! Reduction-specific information of current access
	ReductionInfo *_reductionInfo;

	//! Whether the access is weak
	bool _weak;
	bool _closesReduction;

	//! Reduction stuff
	size_t _reductionLength;
	reduction_type_and_operator_index_t _reductionOperator;
	reduction_index_t _reductionIndex;

	//! Next task with an access matching this one
	std::atomic<Task *> _successor;
	std::atomic<Task *> _child;

	//! Atomic flags for Read / Write / Deletable / Finished
	std::atomic<access_flags_t> _accessFlags;

	Instrument::data_access_id_t _instrumentDataAccessId;
public:
	DataAccess(DataAccessType type, Task *originator, bool weak = false, ReductionInfo * reductionInfo = nullptr)
		: _type(type),
		_originator(originator),
		_reductionInfo(reductionInfo),
		_weak(weak),
		_closesReduction(false),
		_successor(nullptr),
		_child(nullptr),
		_accessFlags(0)
	{
		assert(originator != nullptr);
		assert(!(weak && type == REDUCTION_ACCESS_TYPE));
	}

	DataAccess(const DataAccess &other)
		: _type(other.getType()),
		_originator(other.getOriginator()),
		_reductionInfo(other.getReductionInfo()),
		_weak(other.isWeak()),
		_closesReduction(other.closesReduction()),
		_successor(other.getSuccessor()),
		_child(other.getChild()),
		_accessFlags(other.getFlags())
	{
	}

	~DataAccess()
	{
	}

	inline void setType(DataAccessType type)
	{
		_type = type;
	}

	inline DataAccessType getType() const
	{
		return _type;
	}

	inline Task *getOriginator() const
	{
		return _originator;
	}

	inline bool setClosesReduction(bool value)
	{
		assert(_type == REDUCTION_ACCESS_TYPE && _closesReduction == !value);
		_closesReduction = value;

		if(_closesReduction)
			return _reductionInfo->markAsClosed();

		return false;
	}

	inline bool closesReduction() const
	{
		return _closesReduction;
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

	Task *getSuccessor() const
	{
		return _successor;
	}

	void setSuccessor(Task *successor)
	{
		_successor = successor;
	}

	inline bool isWeak() const
	{
		return _weak;
	}

	inline void setWeak(bool value = true)
	{
		_weak = value;
	}

	void setInstrumentationId(Instrument::data_access_id_t instrumentDataAccessId)
	{
		_instrumentDataAccessId = instrumentDataAccessId;
	}

	Instrument::data_access_id_t & getInstrumentationId()
	{
		return _instrumentDataAccessId;
	}

	size_t getReductionLength() const
	{
		return _reductionLength;
	}

	void setReductionLength(size_t reductionLength)
	{
		_reductionLength = reductionLength;
	}

	reduction_type_and_operator_index_t getReductionOperator() const
	{
		return _reductionOperator;
	}

	void setReductionOperator(reduction_type_and_operator_index_t reductionOperator)
	{
		_reductionOperator = reductionOperator;
	}

	reduction_index_t getReductionIndex() const
	{
		return _reductionIndex;
	}

	void setReductionIndex(reduction_index_t reductionIndex)
	{
		_reductionIndex = reductionIndex;
	}

	inline Task * getChild() const {
		return _child;
	}

	inline void setChild(Task *child) {
		_child = child;
	}

	inline access_flags_t getFlags() const {
		return _accessFlags;
	}

	inline access_flags_t setFlags(access_flags_t flagsToSet) {
		return _accessFlags.fetch_or(flagsToSet);
	}

	inline access_flags_t unsetFlags(access_flags_t flagsToUnset) {
		return _accessFlags.fetch_and(~flagsToUnset);
	}
};


#endif // DATA_ACCESS_HPP
