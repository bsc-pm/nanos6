/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef VIRTUAL_MEMORY_MANAGEMENT_HPP
#define VIRTUAL_MEMORY_MANAGEMENT_HPP

#include <cstdlib>
#include <vector>

#include "DataAccessRegion.hpp"
#include "lowlevel/PaddedSpinLock.hpp"
#include "memory/vmm/VirtualMemoryAllocation.hpp"
#include "memory/vmm/VirtualMemoryArea.hpp"


class MemoryPlace;

class VirtualMemoryManagement {
private:
	//! Default allocation size for OS memory allocations
	static size_t _size;

	//! Memory allocations from OS
	static std::vector<VirtualMemoryAllocation *> _allocations;

	//! Addresses for local NUMA allocations
	typedef std::vector<VirtualMemoryArea *> node_allocations_t;
	static std::vector<node_allocations_t> _localNUMAVMA;

	typedef PaddedSpinLock<> vmm_lock_t;
	static vmm_lock_t _lock;

	//! \brief Set up the memory layout based
	//!
	//! It splits memory allocated from the OS to equally among a collection
	//! of MemoryPlaces
	//!
	//! \param allocation represents the OS memory allocation.
	//! \param[in] nodes is the collections of NUMA nodes among which the allocation will
	//! be divided
	static void setupMemoryLayout(
		VirtualMemoryAllocation *allocation,
		const std::vector<MemoryPlace *> &nodes
	);

	//! \brief Private constructor, this is a singleton
	VirtualMemoryManagement()
	{
	}

public:

	static void initialize();

	static void shutdown();

	//! \brief Allocate a block of generic addresses.
	//! At the moment we redirect this to malloc
	static inline void *allocDistrib(size_t size)
	{
		return malloc(size);
	}

	//! \brief Allocate a block of local addresses on a NUMA node.
	//!
	//! \param[in] size the size to allocate
	//! \param[in] numaNodeId is the the id of the NUMA node to allocate
	static inline void *allocLocalNUMA(size_t size, size_t numaNodeId)
	{
		std::lock_guard<vmm_lock_t> guard(_lock);

		// Try to allocate from the last available memory allocation
		VirtualMemoryArea *vma = _localNUMAVMA[numaNodeId].back();
		void *ret = vma->allocBlock(size);
		if (ret != nullptr) {
			return ret;
		}

		// If allocation from already mapped regions failed create a new mapping
		size_t allocation_size = (size < _size) ? _size : 2 * size;

		// We need the size of allocations to be a multiple of page size
		allocation_size = ROUND_UP(allocation_size, HardwareInfo::getPageSize());

		VirtualMemoryAllocation *alloc = new VirtualMemoryAllocation(
			nullptr, allocation_size);

		_allocations.push_back(alloc);
		_localNUMAVMA[numaNodeId].push_back(
			new VirtualMemoryArea(alloc->getAddress(), allocation_size)
		);

		// This should always succeed
		vma = _localNUMAVMA[numaNodeId].back();
		return vma->allocBlock(size);
	}

	//! \brief Return the NUMA node id of the node containing 'ptr' or
	//! the NUMA node count if not found
	static inline size_t findNUMA(void *ptr)
	{
		std::lock_guard<vmm_lock_t> guard(_lock);
		for (size_t i = 0; i < _localNUMAVMA.size(); ++i) {
			for (auto vma : _localNUMAVMA[i]) {
				if (vma->includesAddress(ptr)) {
					return i;
				}
			}
		}

		//! Non-NUMA allocation
		return _localNUMAVMA.size();
	}

	static inline bool isDistributedRegion(const DataAccessRegion&)
	{
		return false;
	}

	static inline bool isLocalRegion(const DataAccessRegion&)
	{
		return false;
	}

	static inline bool isClusterMemory(const DataAccessRegion&)
	{
		return false;
	}

};


#endif // VIRTUAL_MEMORY_MANAGEMENT_HPP
