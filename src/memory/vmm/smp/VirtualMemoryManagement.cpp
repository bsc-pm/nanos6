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
#include <sys/mman.h>

void *VirtualMemoryManagement::_address;
size_t VirtualMemoryManagement::_size;
std::vector<VirtualMemoryArea *> VirtualMemoryManagement::_localNUMAVMA;
size_t VirtualMemoryManagement::_pageSize;

void VirtualMemoryManagement::initialize()
{
	_pageSize = HardwareInfo::getPageSize();
	size_t totalPhysicalMemory = HardwareInfo::getPhysicalMemorySize();
	
	/* NANOS6_LOCAL_MEMORY determines the size of the local address space
	 * per cluster node.
	 * Default value: Trying to map the minimum between 2GB and the 5% of
	 * 		  the total physical memory of the machine*/
	_size = std::min(2UL << 30, totalPhysicalMemory / 20);
	EnvironmentVariable<StringifiedMemorySize> _sizeEnv("NANOS6_LOCAL_MEMORY", _size);
	_size = _sizeEnv.getValue();
	assert(_size > 0);
	_size = ROUND_UP(_size, _pageSize);
	
	/* At the moment we use as default value for start address 256MB,
	 * because it seems to be working (lower values not always succeed).
	 * TODO: Create a robust protocol that will reduce chances of
	 * failing. */
	EnvironmentVariable<void *> startAddress("NANOS6_VA_START", (void *)(2UL << 27));
	_address = startAddress.getValue();
	
	void *ret = mmap(_address, _size, PROT_READ|PROT_WRITE,
			MAP_ANONYMOUS|MAP_PRIVATE|MAP_NORESERVE, -1, 0);
	FatalErrorHandler::check(
		ret != MAP_FAILED,
		"mapping virtual address space failed"
	);
	
	//! mmap might not have returned the requested address
	_address = ret;
	
	setupMemoryLayout(_address, _size);
	
	RuntimeInfo::addEntry("local_memory_size", "Size of local memory per node", _size);
	RuntimeInfo::addEntry("va_start", "Virtual address space start", (unsigned long)_address);
}

void VirtualMemoryManagement::shutdown()
{
	for (auto &vma : _localNUMAVMA) {
		delete vma;
	}
	
	int ret = munmap(_address, _size);
	FatalErrorHandler::failIf(
		ret == -1,
		"unmapping address space failed"
	);
}

void VirtualMemoryManagement::setupMemoryLayout(void *address, size_t size)
{
	/** We have one VMA per NUMA node. At the moment we divide the local
	 * address space equally among these areas. */
	size_t NUMANodeCount = HardwareInfo::getMemoryNodeCount();
	_localNUMAVMA.resize(NUMANodeCount);
	
	/** Divide the address space between the NUMA nodes and the
	 * making sure that all areas have a size that is multiple
	 * of PAGE_SIZE */
	size_t localPages = size / _pageSize;
	size_t pagesPerNUMA = localPages / NUMANodeCount;
	size_t extraPages = localPages % NUMANodeCount;
	size_t sizePerNUMA = pagesPerNUMA * _pageSize;
	char *ptr = (char *)address;
	for (size_t i = 0; i < NUMANodeCount; ++i) {
		size_t numaSize = sizePerNUMA;
		if (extraPages > 0) {
			numaSize += _pageSize;
			extraPages--;
		}
		_localNUMAVMA[i] = new VirtualMemoryArea(ptr, numaSize);
		ptr += numaSize;
	}
}
