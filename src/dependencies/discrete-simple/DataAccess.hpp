/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_HPP
#define DATA_ACCESS_HPP

#include <atomic>
#include <bitset>
#include <cassert>

#include "../DataAccessType.hpp"
#include "ReductionSpecific.hpp"
#include "ReductionInfo.hpp"
#include <InstrumentDataAccessId.hpp>

struct DataAccess;
class Task;


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

	//! Next task with an access matching this one
	Task * _successor;

	//! Is this the top read access..
	std::atomic<int> _isTop;

	Instrument::data_access_id_t _instrumentDataAccessId;
	
public:
	DataAccess(DataAccessType type, Task *originator, bool weak = false, ReductionInfo * reductionInfo = nullptr)
		: _type(type),
		_originator(originator),
		_reductionInfo(reductionInfo),
		_weak(weak),
		_closesReduction(false),
		_successor(nullptr),
		_isTop(1)
	{
		assert(originator != nullptr);
		assert(!(weak && (type != REDUCTION_ACCESS_TYPE)));
	}
	
	DataAccess(const DataAccess &other)
		: _type(other.getType()),
		_originator(other.getOriginator()),
		_reductionInfo(other.getReductionInfo()),
		_weak(other.isWeak()),
		_closesReduction(other.closesReduction()),
		_successor(other.getSuccessor()),
		_isTop(1)
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
	
	inline bool isReader() const
	{
		return (_type == READ_ACCESS_TYPE || ((_type == REDUCTION_ACCESS_TYPE && !_closesReduction)));
	}
	
	inline bool isWriter() const
	{
		return (_type == WRITE_ACCESS_TYPE || ((_type == REDUCTION_ACCESS_TYPE && _closesReduction)));
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

	Task *getSuccessor() const {
		return _successor;
	}

	void setSuccessor(Task *successor) {
		_successor = successor;
	}

	bool isWeak() const
	{
		return _weak;
	}

	void setInstrumentationId(Instrument::data_access_id_t instrumentDataAccessId) 
	{
		_instrumentDataAccessId = instrumentDataAccessId;
	}

	Instrument::data_access_id_t & getInstrumentationId() 
	{
		return _instrumentDataAccessId;
	}

	inline bool decreaseTop() 
	{
		int res = _isTop.fetch_sub(1, std::memory_order_relaxed);
		assert(res >= 0);
		return (res == 0);
	}

	inline bool isTop() 
	{
		return (_isTop == -1);
	}

	inline bool decreaseReductor() 
	{
		int res = _isTop.fetch_sub(1, std::memory_order_relaxed);
		assert(res >= 0);
		return (res == 0);
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
};


#endif // DATA_ACCESS_HPP
