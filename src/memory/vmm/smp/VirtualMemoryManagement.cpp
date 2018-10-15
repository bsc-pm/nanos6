/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#include "hardware/HardwareInfo.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "memory/vmm/VirtualMemoryArea.hpp"
#include "VirtualMemoryManagement.hpp"
#include "system/RuntimeInfo.hpp"

#include <string>

size_t VirtualMemoryManagement::_size;
std::vector<VirtualMemoryAllocation *> VirtualMemoryManagement::_allocations;
std::vector<VirtualMemoryManagement::node_allocations_t> VirtualMemoryManagement::_localNUMAVMA;
VirtualMemoryManagement::vmm_lock_t VirtualMemoryManagement::_lock;

void VirtualMemoryManagement::initialize()
{
	size_t totalPhysicalMemory = HardwareInfo::getPhysicalMemorySize();
	
	/* NANOS6_LOCAL_MEMORY determines the size of the local address space
	 * per cluster node.
	 * Default value: Trying to map the minimum between 2GB and the 5% of
	 * 		  the total physical memory of the machine*/
	_size = std::min(2UL << 30, totalPhysicalMemory / 20);
	EnvironmentVariable<StringifiedMemorySize> _sizeEnv("NANOS6_LOCAL_MEMORY", _size);
	_size = _sizeEnv.getValue();
	assert(_size > 0);
	_size = ROUND_UP(_size, HardwareInfo::getPageSize());
	
	_allocations.resize(1);
	_allocations[0] = new VirtualMemoryAllocation(nullptr, _size);
	
	_localNUMAVMA.resize(HardwareInfo::getMemoryPlaceCount(nanos6_device_t::nanos6_host_device));
	HostInfo *deviceInfo = (HostInfo *)HardwareInfo::getDeviceInfo(nanos6_device_t::nanos6_host_device);
	setupMemoryLayout(_allocations[0], deviceInfo->getMemoryPlaces());
	
	RuntimeInfo::addEntry("local_memory_size", "Size of local memory per node", _size);
}

void VirtualMemoryManagement::shutdown()
{
	for (auto &map : _localNUMAVMA) {
		for (auto &vma : map) {
			delete vma;
		}
	}
	
	for (auto &allocation : _allocations) {
		delete allocation;
	}
}

void VirtualMemoryManagement::setupMemoryLayout(
		VirtualMemoryAllocation *allocation,
		std::vector<MemoryPlace *> numaNodes)
{
	void *address = allocation->getAddress();
	size_t size = allocation->getSize();
	size_t numaNodeCount = numaNodes.size();
	
	/** Divide the address space between the NUMA nodes and the
	 * making sure that all areas have a size that is multiple
	 * of PAGE_SIZE */
	size_t localPages = size / HardwareInfo::getPageSize();
	size_t pagesPerNUMA = localPages / numaNodeCount;
	size_t extraPages = localPages % numaNodeCount;
	size_t sizePerNUMA = pagesPerNUMA * HardwareInfo::getPageSize();
	char *ptr = (char *)address;
	for (size_t i = 0; i < numaNodeCount; ++i) {
		size_t nodeId = numaNodes[i]->getIndex();
		size_t numaSize = sizePerNUMA;
		if (extraPages > 0) {
			numaSize += HardwareInfo::getPageSize();
			extraPages--;
		}
		_localNUMAVMA[nodeId].push_back(new VirtualMemoryArea(ptr, numaSize));
		ptr += numaSize;
	}
}
