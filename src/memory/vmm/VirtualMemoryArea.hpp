/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef __VIRTUAL_MEMORY_AREA_HPP__
#define __VIRTUAL_MEMORY_AREA_HPP__

#define ROUND_UP(_s, _r) ((((_s) + (_r) - 1) / (_r)) * (_r))

#include "hardware/HardwareInfo.hpp"

#include <sys/mman.h>

class VirtualMemoryArea {
	//! start address of the area
	void *_address;
	
	//! size of the area
	size_t _size;
	
	//! next free pointer in the area
	void *_nextFree;
	
	//! amount of available memory
	size_t _available;
	
public:
	VirtualMemoryArea(void *address, size_t size)
		: _address(address), _size(size),
		_nextFree(address), _available(size)
	{
	}
	
	//! Virtual addresses should be unique, so can't copy this
	VirtualMemoryArea(VirtualMemoryArea const &) = delete;
	VirtualMemoryArea operator=(VirtualMemoryArea const &) = delete;
	
	~VirtualMemoryArea()
	{
	}
	
	/** Returns a block of virtual address from the virtual memory area.
	 *
	 * This method is not thread-safe. Synchronization needs to be handled
	 * externaly. */
	inline void *allocBlock(size_t size)
	{
		/** Rounding up the size allocations to PAGE_SIZE is the easy
		 * way to ensure all allocations are aligned to PAGE_SIZE */
		size = ROUND_UP(size, HardwareInfo::getPageSize());
		if (size > _available) {
			return nullptr;
		}
		
		void *ret = _nextFree;
		
		_available -= size;
		_nextFree = (void *)((char *)_nextFree + size);
		
		return ret;
	}
};

#endif /* __VIRTUAL_MEMORY_AREA_HPP__ */
