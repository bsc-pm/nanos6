/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef MEMORY_PLACE_HPP
#define MEMORY_PLACE_HPP

#include <vector>
#include <map>
#include "memory/AddressSpace.hpp"

#include <nanos6/task-instantiation.h>

class ComputePlace;

class MemoryPlace {
protected:
	AddressSpace * _addressSpace;
	int _index;	
	nanos6_device_t _type;	
	
public:
	MemoryPlace(int index, nanos6_device_t type, AddressSpace * addressSpace = nullptr)
		: _addressSpace(addressSpace), _index(index), _type(type)
	{}
	
	virtual ~MemoryPlace() 
	{}
	
	inline int getIndex(void) const
	{ 
		return _index; 
	} 
	
	inline nanos6_device_t getType() const
	{
		return _type;
	}
	
	inline AddressSpace * getAddressSpace() const
	{ 
		return _addressSpace; 
	} 
};

#endif //MEMORY_PLACE_HPP
