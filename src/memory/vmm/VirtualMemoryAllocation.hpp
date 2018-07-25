/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/


#ifndef __VIRTUAL_MEMORY_ALLOCATION_HPP__
#define __VIRTUAL_MEMORY_ALLOCATION_HPP__

#include "lowlevel/FatalErrorHandler.hpp"
#include <sys/mman.h>


class VirtualMemoryAllocation {
	//! first address of the allocation
	void *_address;
	
	//! size of the allocation
	size_t _size;
	
	void unmap(void *addr, size_t size)
	{
		int ret = munmap(addr, size);
		FatalErrorHandler::failIf(
			ret != 0,
			"Could not unmap memory allocation"
		);
	}
	
public:
	VirtualMemoryAllocation(void *address, size_t size)
		: _address(address), _size(size)
	{
		/** For the moment we are using fixed memory protection and
		 * allocation flags, but in the future we could make those
		 * arguments fields of the class */
		int prot = PROT_READ|PROT_WRITE;
		int flags = MAP_ANONYMOUS|MAP_PRIVATE|MAP_NORESERVE;
		if (_address != nullptr) {
			flags |= MAP_FIXED;
		}
		void *ret = mmap(_address, _size, prot, flags, -1, 0);
		FatalErrorHandler::check(
			ret != MAP_FAILED,
			"mapping virtual address space failed"
		);
		
		_address = ret;
	}
	
	~VirtualMemoryAllocation()
	{
		unmap(_address, _size);
	}
	
	inline void *getAddress() const
	{
		return _address;
	}
	
	inline size_t getSize() const
	{
		return _size;
	}
};

#endif /* __VIRTUAL_MEMORY_ALLOCATION_HPP__ */
