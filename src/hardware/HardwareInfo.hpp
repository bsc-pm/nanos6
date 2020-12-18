/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef HARDWARE_INFO_HPP
#define HARDWARE_INFO_HPP

#include <vector>

#include "hwinfo/DeviceInfo.hpp"
#include "hwinfo/HostInfo.hpp"
#include "places/ComputePlace.hpp"
#include "places/MemoryPlace.hpp"
#include "places/HardwareCache.hpp"

class HardwareInfo {
private:
	static std::vector<DeviceInfo *> _infos;

public:

	static void initialize();

	static void shutdown();

	static inline bool canDeviceRunTasks(nanos6_device_t type)
	{
		if (_infos[type] == nullptr)
			return false;

		return _infos[type]->isDeviceInitialized();
	}

	static inline size_t getComputePlaceCount(nanos6_device_t type)
	{
		return _infos[type]->getComputePlaceCount();
	}

	static inline ComputePlace *getComputePlace(nanos6_device_t type, int index)
	{
		return _infos[type]->getComputePlace(index);
	}

	static inline size_t getValidMemoryPlaceCount(nanos6_device_t type)
	{
		if (type == nanos6_host_device) {
			HostInfo *hostInfo = (HostInfo *)_infos[type];
			return hostInfo->getValidMemoryPlaceCount();
		}

		return _infos[type]->getMemoryPlaceCount();
	}

	static inline size_t getMemoryPlaceCount(nanos6_device_t type)
	{
		return _infos[type]->getMemoryPlaceCount();
	}

	static inline MemoryPlace *getMemoryPlace(nanos6_device_t type, int index)
	{
		return _infos[type]->getMemoryPlace(index);
	}

	static DeviceInfo *getDeviceInfo(nanos6_device_t type)
	{
		return _infos[type];
	}

	static inline size_t getCacheLineSize()
	{
		return ((HostInfo *) _infos[nanos6_device_t::nanos6_host_device])->getCacheLineSize();
	}

	static inline size_t getPageSize()
	{
		return ((HostInfo *) _infos[nanos6_device_t::nanos6_host_device])->getPageSize();
	}

	static inline size_t getPhysicalMemorySize()
	{
		return ((HostInfo *) _infos[nanos6_device_t::nanos6_host_device])->getPhysicalMemorySize();
	}

	static inline size_t getNumPhysicalPackages()
	{
		return ((HostInfo *) _infos[nanos6_device_t::nanos6_host_device])->getNumPhysicalPackages();
    }

	static inline size_t getNumL2Cache()
	{
		return ((HostInfo *) _infos[nanos6_device_t::nanos6_host_device])->getNumL2Cache();
	}

	static inline size_t getNumL3Cache()
	{
		return ((HostInfo *) _infos[nanos6_device_t::nanos6_host_device])->getNumL3Cache();
	}

	static inline L2Cache *getL2Cache(DataTrackingSupport::location_t loc)
	{
		return ((HostInfo *) _infos[nanos6_device_t::nanos6_host_device])->getL2Cache(loc);
	}

	static inline L3Cache *getL3Cache(DataTrackingSupport::location_t loc)
	{
		return ((HostInfo *) _infos[nanos6_device_t::nanos6_host_device])->getL3Cache(loc);
	}
};

#endif // HARDWARE_INFO_HPP
