/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NUMA_PLACE_HPP
#define NUMA_PLACE_HPP

#include "MemoryPlace.hpp"

class NUMAPlace: public MemoryPlace {
private:
	typedef std::map<int, ComputePlace*> computePlaces_t;
	computePlaces_t _computePlaces; //ComputePlaces able to interact with this MemoryPlace
public:
	NUMAPlace(int index, AddressSpace * addressSpace = nullptr)
		: MemoryPlace(index, nanos6_device_t::nanos6_host_device, addressSpace)
	{}
	
	virtual ~NUMAPlace() {}
	size_t getComputePlaceCount(void) const { return _computePlaces.size(); }
	ComputePlace* getComputePlace(int index){ return _computePlaces[index]; }
	void addComputePlace(ComputePlace* computePlace);
	std::vector<int> getComputePlaceIndexes();
	std::vector<ComputePlace*> getComputePlaces();
};

#endif //NUMA_PLACE_HPP
