/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef HOST_INFO_HPP
#define HOST_INFO_HPP

#include "DeviceInfo.hpp"
#include "dependencies/DataTrackingSupport.hpp"

class L2Cache;
class L3Cache;

class HostInfo : public DeviceInfo {
private:

	//! List of CPUs on the system
	std::vector<ComputePlace *> _computePlaces;

	//! List of NUMA nodes on the system
	std::vector<MemoryPlace *> _memoryPlaces;

	//!< List of L2 caches on the system
	std::vector<L2Cache*> _L2Caches;

	//!< List of L3 caches on the system
	std::vector<L3Cache*> _L3Caches;

	//! L1 Cache line size
	size_t _cacheLineSize;

	//! Page size of the system
	size_t _pageSize;

	//! Total amount of physical memory on the system
	size_t _physicalMemorySize;

	//! Total amount of valid memory places in the system
	size_t _validMemoryPlaces;

	//! Total amount of physical packages in the system
	size_t _numPhysicalPackages;

	//! Matrix of NUMA distances
	std::vector<uint64_t> _NUMADistances;

public:

	HostInfo();

	~HostInfo();

	inline size_t getComputePlaceCount() const override
	{
		return _computePlaces.size();
	}

	inline ComputePlace *getComputePlace(int index) const override
	{
		return _computePlaces[index];
	}

	inline const std::vector<ComputePlace *> &getComputePlaces() const
	{
		return _computePlaces;
	}

	inline size_t getMemoryPlaceCount() const override
	{
		return _memoryPlaces.size();
	}

	inline size_t getValidMemoryPlaceCount() const
	{
		return _validMemoryPlaces;
	}

	inline MemoryPlace *getMemoryPlace(int index) const override
	{
		return _memoryPlaces[index];
	}

	inline const std::vector<MemoryPlace *> &getMemoryPlaces() const
	{
		return _memoryPlaces;
	}

	inline size_t getCacheLineSize() const
	{
		return _cacheLineSize;
	}

	inline size_t getPageSize() const
	{
		return _pageSize;
	}

	inline size_t getPhysicalMemorySize() const
	{
		return _physicalMemorySize;
	}

	inline size_t getNumPhysicalPackages() const override
	{
		return _numPhysicalPackages;
	}
	inline size_t getNumL2Cache() const
	{
		return _L2Caches.size();
	}

	inline size_t getNumL3Cache() const
	{
		return _L3Caches.size();
	}

	inline L2Cache *getL2Cache(DataTrackingSupport::location_t loc) const
	{
		return _L2Caches[loc];
	}

	inline L3Cache *getL3Cache(DataTrackingSupport::location_t loc) const
	{
		return _L3Caches[loc];
	}

	inline const std::vector<uint64_t> &getNUMADistances() const
	{
		return _NUMADistances;
	}
};

#endif // HOST_INFO_HPP
