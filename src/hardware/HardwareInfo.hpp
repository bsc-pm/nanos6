/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef HARDWARE_INFO_HPP
#define HARDWARE_INFO_HPP

#include <vector>

#include "hwinfo/DeviceInfo.hpp"
#include "hwinfo/HostInfo.hpp"
#include "places/ComputePlace.hpp"
#include "places/MemoryPlace.hpp"

class HardwareInfo {
private:
	static std::vector<DeviceInfo *> _infos;

public:

	static void initialize();

	static void initializeDeviceServices();

	static void shutdown();

	static void shutdownDeviceServices();

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

	static inline const std::vector<uint64_t> &getNUMADistances()
	{
		return ((HostInfo *) _infos[nanos6_device_t::nanos6_host_device])->getNUMADistances();
	}
};

#endif // HARDWARE_INFO_HPP
