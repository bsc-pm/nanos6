/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef __VIRTUAL_MEMORY_MANAGEMENT_HPP__
#define __VIRTUAL_MEMORY_MANAGEMENT_HPP__

#include "memory/vmm/VirtualMemoryArea.hpp"

#include <vector>

class VirtualMemoryAllocation;

class VirtualMemoryManagement {
private:
	//! memory allocations from OS
	static std::vector<VirtualMemoryAllocation *> _allocations;
	
	//! addresses for local NUMA allocations
	static std::vector<VirtualMemoryArea *> _localNUMAVMA;
	
	//! addresses for generic allocations
	static VirtualMemoryArea *_genericVMA;
	
	//! Setting up the memory layout
	static void setupMemoryLayout(void *address, size_t distribSize, size_t localSize);
	
	//! private constructor, this is a singleton.
	VirtualMemoryManagement()
	{
	}
public:
	static void initialize();
	static void shutdown();
	
	/** allocate a block of generic addresses.
	 *
	 * This region is meant to be used for allocations that can be mapped
	 * to various memory nodes (cluster or NUMA) based on a policy. So this
	 * is the pool for distributed allocations or other generic allocations.
	 */
	static inline void *allocDistrib(size_t size)
	{
		return _genericVMA->allocBlock(size);
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
	
	//! return the NUMA node id of the node containing 'ptr' or
	//! the NUMA node count if not found
	static inline size_t findNUMA(void *ptr)
	{
		for (size_t i = 0; i < _localNUMAVMA.size(); ++i) {
			if (_localNUMAVMA[i]->includesAddress(ptr)) {
				return i;
			}
		}
		
		//! Non-NUMA allocation
		return _localNUMAVMA.size();
	}
	
	//! check if a region is within the distributed memory regions
	static inline bool isDistributedRegion(DataAccessRegion const &region)
	{
		return _genericVMA->includesRange(region.getStartAddress(),
				region.getSize());
	}
};


#endif /* __VIRTUAL_MEMORY_MANAGEMENT_HPP__ */
