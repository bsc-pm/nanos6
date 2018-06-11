/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef COMPUTE_PLACE_HPP
#define COMPUTE_PLACE_HPP

#include <InstrumentComputePlaceId.hpp>
#include "CPUDependencyData.hpp"

#include <map>
#include <vector>

#include <nanos6/task-instantiation.h>

class MemoryPlace;

//! \brief A class that represents a place where code can be executed either directly, or in a sub-place within
class ComputePlace {
private:
	typedef std::map<int, MemoryPlace*> memoryPlaces_t;
	memoryPlaces_t _memoryPlaces; // Accessible MemoryPlaces from this ComputePlace 

protected:
	//ComputePlace * _parent;
	int _index;
	nanos6_device_t _type;	

	Instrument::compute_place_id_t _instrumentationId;

	CPUDependencyData _dependencyData;

public:
	void *_schedulerData;
	
	ComputePlace(int index, nanos6_device_t type)
		: _index(index), _type(type), _schedulerData(nullptr)
	{}
	
	virtual ~ComputePlace() 
	{}
	
	size_t getMemoryPlacesCount(void) const 
	{ 
		return _memoryPlaces.size(); 
	}
	
	MemoryPlace* getMemoryPlace(int index) 
	{ 
		memoryPlaces_t::iterator it = _memoryPlaces.find(index);
		if (it != _memoryPlaces.end()) {
			return it->second;
		}
		return nullptr;
	}
	
	inline int getIndex(void) const 
	{ 
		return _index; 
	} 

	inline nanos6_device_t getType() 
	{
		return _type;
	}	

	void addMemoryPlace(MemoryPlace* mem);
	std::vector<int> getMemoryPlacesIndexes();
	std::vector<MemoryPlace*> getMemoryPlaces();

	void setInstrumentationId(Instrument::compute_place_id_t const &instrumentationId)
	{
		_instrumentationId = instrumentationId;
	}
	
	Instrument::compute_place_id_t const &getInstrumentationId() const
	{
		return _instrumentationId;
	}
	
	CPUDependencyData &getDependencyData()
	{
		return _dependencyData;
	}
	
};

#endif //COMPUTE_PLACE_HPP
