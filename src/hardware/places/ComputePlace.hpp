/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef COMPUTE_PLACE_HPP
#define COMPUTE_PLACE_HPP


#include <map>
#include <vector>

#include <nanos6/task-instantiation.h>

#include "CPUDependencyData.hpp"

#include <InstrumentComputePlaceId.hpp>

class Taskfor;
class MemoryPlace;

//! \brief A class that represents a place where code can be executed either directly, or in a sub-place within
class ComputePlace {
private:
	typedef std::map<int, MemoryPlace *> memory_places_t;
	memory_places_t _memoryPlaces; // Accessible MemoryPlaces from this ComputePlace
	//! Preallocated taskfor to be used as taskfor collaborator.
	Taskfor *_preallocatedTaskfor;
	//! Preallocated argsBlock to be used for taskfor collaborators.
	void *_preallocatedArgsBlock;
	size_t _preallocatedArgsBlockSize;
	
protected:
	size_t _index;
	nanos6_device_t _type;
	
	Instrument::compute_place_id_t _instrumentationId;
	
	CPUDependencyData _dependencyData;
	
public:
	void *_schedulerData;
	
	ComputePlace(int index, nanos6_device_t type);
	
	virtual ~ComputePlace();
	
	size_t getMemoryPlacesCount() const
	{
		return _memoryPlaces.size();
	}
	
	MemoryPlace *getMemoryPlace(int index)
	{
		memory_places_t::iterator it = _memoryPlaces.find(index);
		if (it != _memoryPlaces.end()) {
			return it->second;
		}
		return nullptr;
	}
	
	//! \brief returns the preallocated taskfor of this ComputePlace.
	inline Taskfor *getPreallocatedTaskfor()
	{
		return _preallocatedTaskfor;
	}
	
	void *getPreallocatedArgsBlock(size_t requiredSize);
	
	inline int getIndex() const
	{
		return _index;
	}
	
	inline void setIndex(int index)
	{
		_index = index;
	}
	
	inline nanos6_device_t getType()
	{
		return _type;
	}
	
	void addMemoryPlace(MemoryPlace *mem);
	
	std::vector<int> getMemoryPlacesIndexes();
	
	std::vector<MemoryPlace *> getMemoryPlaces();
	
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
