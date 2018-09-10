/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
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
	
public:
	void initialize();
	void shutdown();
	
	inline size_t getComputePlaceCount(void) 
	{
		return _computePlaces.size();
	}
	inline ComputePlace* getComputePlace(int index) 
	{
		return _computePlaces[index];
	}
	inline std::vector<ComputePlace *> getComputePlaces()
	{
		return _computePlaces;
	}
	
	inline size_t getMemoryPlaceCount(void)
	{
		return _memoryPlaces.size();
	}
	inline MemoryPlace* getMemoryPlace(int index)
	{
		return _memoryPlaces[index];
	}
	inline std::vector<MemoryPlace *> getMemoryPlaces()
	{
		return _memoryPlaces;
	}
	
	inline size_t getCacheLineSize(void)
	{
		return _cacheLineSize;
	}
	inline size_t getPageSize(void)
	{
		return _pageSize;
	}
	inline size_t getPhysicalMemorySize(void)
	{
		return _physicalMemorySize;
	}
};

#endif //HOST_INFO_HPP
