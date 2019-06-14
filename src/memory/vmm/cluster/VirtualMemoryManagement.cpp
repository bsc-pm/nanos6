/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#include <string>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>

#include "VirtualMemoryManagement.hpp"
#include "cluster/ClusterManager.hpp"
#include "cluster/messages/MessageId.hpp"
#include "hardware/HardwareInfo.hpp"
#include "hardware/cluster/ClusterNode.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "memory/vmm/VirtualMemoryAllocation.hpp"
#include "memory/vmm/VirtualMemoryArea.hpp"
#include "system/RuntimeInfo.hpp"

#include <DataAccessRegion.hpp>
#include <Directory.hpp>

std::vector<VirtualMemoryAllocation *> VirtualMemoryManagement::_allocations;
std::vector<VirtualMemoryArea *> VirtualMemoryManagement::_localNUMAVMA;
VirtualMemoryArea *VirtualMemoryManagement::_genericVMA;


//! \brief Returns a vector with all the mappings of the process
//!
//! It parses /proc/self/maps to find out all the current mappings of the
//! process.
//!
//! \returns a vector of DataAccessRegion objects describing the mappings
static std::vector<DataAccessRegion> findMappedRegions()
{
	std::vector<DataAccessRegion> maps;
	const char *mappingsFile = "/proc/self/maps";
	size_t len = 0;
	char *line = NULL;
	
	FILE *fp = fopen(mappingsFile, "r");
	assert(fp != NULL);
	
	ssize_t ret = getline(&line, &len, fp);
	FatalErrorHandler::failIf(ret == -1, "Could not find virtual memory mappings");
	
	while (ret != -1) {
		// First, we take the range which appears first to the line
		// separated by a space with everything that follows
		char *token = strtok(line, " ");
		assert(token != NULL);
		
		// Then we need to split the range which is two hexadecimals
		// separated by a '-'
		token = strtok(token, "-");
		void *startAddress = (void *)strtoll(token, NULL, 16);
		token = strtok(NULL, "-");
		void *endAddress = (void *)strtoll(token, NULL, 16);
		
		// The lower-end of the canonical virtual addresses finish
		// at the 2^47 limit. The upper-end of the canonical addresses
		// are normally used by the linux kernel. So we don't want to
		// look there.
		if ((size_t)endAddress >= (1UL << 47)) {
			break;
		}
		
		maps.emplace_back(startAddress, endAddress);
		
		// Read next line
		ret = getline(&line, &len, fp);
	}
	
	fclose(fp);
	
	return maps;
}

//! \brief Finds an memory region to map the Nanos6 region
//!
//! This finds the biggest common gap (region not currently mapped in the
//! virtual address space) of all Nanos6 instances and returns it.
//!
//! \returns an available memory region to map Nanos6 memory, or an empty region
//!          if none available
static DataAccessRegion findSuitableMemoryRegion()
{
	std::vector<DataAccessRegion> maps = findMappedRegions();
	size_t length = maps.size();
	DataAccessRegion gap;
	
	//! Find the biggest gap locally
	for (size_t i = 1; i < length; ++i) {
		void *previousEnd = maps[i - 1].getEndAddress();
		void *nextStart = maps[i].getStartAddress();
		
		DataAccessRegion region(previousEnd, nextStart);
		if (region.getSize() > gap.getSize()) {
			gap = region;
		}
	}
	
	//! If not in cluster mode, we are done here
	if (!ClusterManager::inClusterMode()) {
		return gap;
	}
	
	int messageId = MessageId::nextMessageId();
	if (ClusterManager::isMasterNode()) {
		// Master node gathers all the gaps from all other nodes and
		// calculates the intersection of all those.
		DataAccessRegion remoteGap;
		DataAccessRegion buffer(&remoteGap, sizeof(remoteGap));
		
		std::vector<ClusterNode *> const &nodes =
			ClusterManager::getClusterNodes();
		
		for (ClusterNode *remote : nodes) {
			if (remote == ClusterManager::getCurrentClusterNode()) {
				continue;
			}
			
			MemoryPlace *memoryNode = remote->getMemoryNode();
			ClusterManager::fetchDataRaw(buffer, memoryNode,
					messageId, true);
			
			gap = gap.intersect(remoteGap);
		}
		
		// Finally, it send the common gap to all other nodes.
		remoteGap = gap;
		for (ClusterNode *remote : nodes) {
			if (remote == ClusterManager::getCurrentClusterNode()) {
				continue;
			}
			
			MemoryPlace *memoryNode = remote->getMemoryNode();
			ClusterManager::sendDataRaw(buffer, memoryNode,
					messageId, true);
		}
	} else {
		DataAccessRegion buffer(&gap, sizeof(gap));
		ClusterNode *master = ClusterManager::getMasterNode();
		MemoryPlace *masterMemory = master->getMemoryNode();
		
		// First send my local gap to master node
		ClusterManager::sendDataRaw(buffer, masterMemory, messageId,
				true);
		
		// Then receive the intersection of all gaps
		ClusterManager::fetchDataRaw(buffer, masterMemory, messageId,
				true);
	}
	
	return gap;
}

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
	
	DataAccessRegion gap = findSuitableMemoryRegion();
	void *address = gap.getStartAddress();
	size_t size = distribSize + localSize * ClusterManager::clusterSize();
	
	FatalErrorHandler::failIf(gap.getSize() < size,
				"Cannot allocate virtual memory region");
	
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
	
	/* Register local addresses with the Directory */
	for (int i = 0; i < clusterSize; ++i) {
		if (i == nodeIndex) {
			continue;
		}
		
		void *ptr = (void *)((char *)address + i * localSize);
		DataAccessRegion localRegion(ptr, localSize);
		Directory::insert(localRegion, ClusterManager::getMemoryNode(i));
	}
	
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
		
		//! Register the region with the Directory
		DataAccessRegion numaRegion(ptr, numaSize);
		Directory::insert(numaRegion, HardwareInfo::getMemoryPlace(
					nanos6_host_device, i));
		
		ptr += numaSize;
	}
}
