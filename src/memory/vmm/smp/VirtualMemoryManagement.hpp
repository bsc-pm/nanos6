/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef __VIRTUAL_MEMORY_MANAGEMENT_HPP__
#define __VIRTUAL_MEMORY_MANAGEMENT_HPP__

#include "memory/vmm/VirtualMemoryArea.hpp"

#include <vector>
#include <stdlib.h>

class VirtualMemoryManagement {
private:
	//! initial allocation from OS
	static void *_address;
	static size_t _size;
	
	//! System's page size
	static size_t _pageSize;
	
	//! addresses for local NUMA allocations
	static std::vector<VirtualMemoryArea *> _localNUMAVMA;
	
	//! Setting up the memory layout
	static void setupMemoryLayout(void *address, size_t size);
	
	//! private constructor, this is a singleton.
	VirtualMemoryManagement()
	{
	}
public:
	static void initialize();
	static void shutdown();
	
	/** allocate a block of generic addresses.
	 *
	 * At the moment we redirect this to malloc
	 */
	static inline void *allocDistrib(size_t size)
	{
		return malloc(size);
	}
	
	/** allocate a block of local addresses on a NUMA node.
	 *
	 * \param size the size to allocate
	 * \param NUMAId is the the id of the NUMA node to allocate
	 */
	static inline void *allocLocalNUMA(size_t size, size_t NUMAId)
	{
		VirtualMemoryArea *vma = _localNUMAVMA.at(NUMAId);
		return vma->allocBlock(size);
	}
};


#endif /* __VIRTUAL_MEMORY_MANAGEMENT_HPP__ */
