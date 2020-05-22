/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef HOST_INFO_HPP
#define HOST_INFO_HPP

#include "DeviceInfo.hpp"

class HostInfo: public DeviceInfo {
private:
	std::vector<ComputePlace *> _computePlaces; //!< List of CPUs on the system
	std::vector<MemoryPlace *> _memoryPlaces;	//!< List of NUMA nodes on the system

	size_t _cacheLineSize;						//!< L1 Cache line size
	size_t _pageSize;							//!< Page size of the system
	size_t _physicalMemorySize;					//!< Total amount of physical memory on the system
	size_t _validMemoryPlaces;

public:
	HostInfo();
	~HostInfo();

	inline size_t getComputePlaceCount() const
	{
		return _computePlaces.size();
	}

	inline ComputePlace *getComputePlace(int index)
	{
		return _computePlaces[index];
	}

	inline std::vector<ComputePlace *> getComputePlaces()
	{
		return _computePlaces;
	}

	inline size_t getMemoryPlaceCount() const
	{
		return _memoryPlaces.size();
	}

	inline size_t getValidMemoryPlaceCount() const
	{
		return _validMemoryPlaces;
	}

	inline MemoryPlace *getMemoryPlace(int index)
	{
		return _memoryPlaces[index];
	}

	inline std::vector<MemoryPlace *> getMemoryPlaces()
	{
		return _memoryPlaces;
	}

	inline size_t getCacheLineSize()
	{
		return _cacheLineSize;
	}

	inline size_t getPageSize()
	{
		return _pageSize;
	}

	inline size_t getPhysicalMemorySize()
	{
		return _physicalMemorySize;
	}
};

#endif // HOST_INFO_HPP
