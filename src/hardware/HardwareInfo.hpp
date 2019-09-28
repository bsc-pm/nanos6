/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef HARDWARE_INFO_HPP
#define HARDWARE_INFO_HPP

#include <map>
#include <vector>

#include "hwinfo/DeviceInfo.hpp"
#include "hwinfo/HostInfo.hpp"
#include "places/ComputePlace.hpp"
#include "places/MemoryPlace.hpp"

class DeviceFunctionsInterface;

class HardwareInfo {
private:
	static std::vector<DeviceInfo *> _infos;
	static std::vector<DeviceFunctionsInterface *> _functions;
public:
	
	static void initialize();
	static void shutdown();
	
	static bool canDeviceRunTasks(nanos6_device_t type);
	
	static inline DeviceFunctionsInterface *getDeviceFunctions(nanos6_device_t type)
	{
		if (type >= _functions.size())
			return nullptr;
		return _functions[type];
	}
	
	static inline size_t getComputePlaceCount(nanos6_device_t type)
	{
		return _infos[type]->getComputePlaceCount();
	}
	static inline ComputePlace *getComputePlace(nanos6_device_t type, int index)
	{
		return _infos[type]->getComputePlace(index);
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
};

#endif // HARDWARE_INFO_HPP
