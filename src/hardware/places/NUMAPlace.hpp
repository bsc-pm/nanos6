/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef NUMA_PLACE_HPP
#define NUMA_PLACE_HPP

#include "MemoryPlace.hpp"

class NUMAPlace: public MemoryPlace {
	typedef std::map<int, ComputePlace *> compute_places_t;
	compute_places_t _computePlaces; //ComputePlaces able to interact with this MemoryPlace
	
public:
	NUMAPlace(int index, AddressSpace *addressSpace = nullptr)
		: MemoryPlace(index, nanos6_device_t::nanos6_host_device, addressSpace)
	{}
	
	virtual ~NUMAPlace()
	{}
	
	inline size_t getComputePlaceCount() const
	{
		return _computePlaces.size();
	}
	
	inline ComputePlace *getComputePlace(int index)
	{
		return _computePlaces[index];
	}
	
	void addComputePlace(ComputePlace *computePlace);
	
	std::vector<int> getComputePlaceIndexes();
	
	std::vector<ComputePlace *> getComputePlaces();
};

#endif //NUMA_PLACE_HPP
