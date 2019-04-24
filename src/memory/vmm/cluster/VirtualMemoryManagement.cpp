/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#include "hardware/HardwareInfo.hpp"
#include "hardware/cluster/ClusterNode.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "memory/vmm/VirtualMemoryAllocation.hpp"
#include "memory/vmm/VirtualMemoryArea.hpp"
#include "VirtualMemoryManagement.hpp"
#include "system/RuntimeInfo.hpp"
#include "cluster/ClusterManager.hpp"

#include <string>
#include <sys/mman.h>

std::vector<VirtualMemoryAllocation *> VirtualMemoryManagement::_allocations;
std::vector<VirtualMemoryArea *> VirtualMemoryManagement::_localNUMAVMA;
VirtualMemoryArea *VirtualMemoryManagement::_genericVMA;

void VirtualMemoryManagement::initialize()
{
	size_t totalPhysicalMemory = HardwareInfo::getPhysicalMemorySize();
	size_t distribSize = 0;
	
	/* NANOS6_DISTRIBUTED_MEMORY determines the total address space to be
	 * used for distributed allocations across the cluster.
	 * Default value: 2GB */
	EnvironmentVariable<StringifiedMemorySize> distribSizeEnv("NANOS6_DISTRIBUTED_MEMORY", (2UL << 30));
	distribSize = distribSizeEnv.getValue();
	assert(distribSize > 0);
	distribSize = ROUND_UP(distribSize, HardwareInfo::getPageSize());
	
	/* NANOS6_LOCAL_MEMORY determines the size of the local address space
	 * per cluster node.
	 * Default value: Trying to map the minimum between 2GB and the 5% of
	 * 		  the total physical memory of the machine*/
	size_t localSize = std::min(2UL << 30, totalPhysicalMemory / 20);
	EnvironmentVariable<StringifiedMemorySize> localSizeEnv("NANOS6_LOCAL_MEMORY", localSize);
	localSize = localSizeEnv.getValue();
	assert(localSize > 0);
	localSize = ROUND_UP(localSize, HardwareInfo::getPageSize());
	
	/* At the moment we use as default value for start address 256MB,
	 * because it seems to be working (lower values not always succeed).
	 * TODO: Create a robust protocol that will reduce chances of
	 * failing. */
	EnvironmentVariable<void *> startAddress("NANOS6_VA_START", (void *)(2UL << 27));
	void *address = startAddress.getValue();
	
	size_t size = distribSize + localSize * ClusterManager::clusterSize();
	
	_allocations.resize(1);
	_allocations[0] = new VirtualMemoryAllocation(address, size);
	
	setupMemoryLayout(address, distribSize, localSize);
	
	RuntimeInfo::addEntry("distributed_memory_size", "Size of distributed memory", distribSize);
	RuntimeInfo::addEntry("local_memorysize", "Size of local memory per node", localSize);
	RuntimeInfo::addEntry("va_start", "Virtual address space start", (unsigned long)address);
}

void VirtualMemoryManagement::shutdown()
{
	for (auto &vma : _localNUMAVMA) {
		delete vma;
	}
	delete _genericVMA;
	
	for (auto &alloc : _allocations) {
		delete alloc;
	}
}

void VirtualMemoryManagement::setupMemoryLayout(void *address, size_t distribSize, size_t localSize)
{
	ClusterNode *current = ClusterManager::getCurrentClusterNode();
	int nodeIndex = current->getIndex();
	int clusterSize = ClusterManager::clusterSize();
	
	void *distribAddress = (void *)((char *)address + clusterSize * localSize);
	_genericVMA = new VirtualMemoryArea(distribAddress, distribSize);
	void *localAddress = (void *)((char *)address + nodeIndex * localSize);
	
	/** We have one VMA per NUMA node. At the moment we divide the local
	 * address space equally among these areas. */
	size_t numaNodeCount = HardwareInfo::getMemoryPlaceCount(nanos6_device_t::nanos6_host_device);
	_localNUMAVMA.resize(numaNodeCount);
	
	/** Divide the address space between the NUMA nodes and the
	 * making sure that all areas have a size that is multiple
	 * of PAGE_SIZE */
	size_t localPages = localSize / HardwareInfo::getPageSize();
	size_t pagesPerNUMA = localPages / numaNodeCount;
	size_t extraPages = localPages % numaNodeCount;
	size_t sizePerNUMA = pagesPerNUMA * HardwareInfo::getPageSize();
	char *ptr = (char *)localAddress;
	for (size_t i = 0; i < numaNodeCount; ++i) {
		size_t numaSize = sizePerNUMA;
		if (extraPages > 0) {
			numaSize += HardwareInfo::getPageSize();
			extraPages--;
		}
		_localNUMAVMA[i] = new VirtualMemoryArea(ptr, numaSize);
		ptr += numaSize;
	}
}
