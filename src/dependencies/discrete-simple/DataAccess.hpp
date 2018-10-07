/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_HPP
#define DATA_ACCESS_HPP

#include <atomic>
#include <bitset>
#include <cassert>
#include <set>

#include "../DataAccessType.hpp"

struct DataAccess;
class Task;


//! The accesses that one or more tasks perform sequentially to a memory location that can occur concurrently (unless commutative).
struct DataAccess {
private:
	//! The type of the access
	DataAccessType _type;
	
	//! The originator of the access
	Task *_originator;
	
public:
	DataAccess(DataAccessType type, Task *originator)
		: _type(type),
		_originator(originator)
	{
		assert(originator != nullptr);
	}
	
	DataAccess(const DataAccess &other)
		: _type(other.getType()),
		_originator(other.getOriginator())
	{}
	
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
		return (_type == READ_ACCESS_TYPE);
	}
	
	inline bool isWriter() const
	{
		return (_type != READ_ACCESS_TYPE);
	}
	
	inline Task *getOriginator() const
	{
		return _originator;
	}
};


#endif // DATA_ACCESS_HPP
