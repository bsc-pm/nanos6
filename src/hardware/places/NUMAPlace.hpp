/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef NUMA_PLACE_HPP
#define NUMA_PLACE_HPP

#include "MemoryPlace.hpp"

class NUMAPlace: public MemoryPlace {
	typedef std::map<int, ComputePlace *> compute_places_t;
	compute_places_t _computePlaces; //ComputePlaces able to interact with this MemoryPlace
	size_t _localCores; // cores that belong to this NUMA node

public:
	NUMAPlace(int index, AddressSpace *addressSpace = nullptr)
		: MemoryPlace(index, nanos6_device_t::nanos6_host_device, addressSpace),
		  _localCores(0)
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

	inline size_t getLocalCoreCount()
	{
		return _localCores;
	}

	inline void incrementLocalCoreCount()
	{
		_localCores++;
	}

	void addComputePlace(ComputePlace *computePlace);

	std::vector<int> getComputePlaceIndexes();

	std::vector<ComputePlace *> getComputePlaces();
};

#endif //NUMA_PLACE_HPP
