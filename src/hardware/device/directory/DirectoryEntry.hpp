/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023-2024 Barcelona Supercomputing Center (BSC)
*/

#ifndef DIRECTORY_ENTRY_HPP
#define DIRECTORY_ENTRY_HPP

#include <cassert>

#include "DirectoryPage.hpp"

class DirectoryEntry
{
	// Base (real) address for the allocation. This refers to the allocation (which must be contiguous) in the home
	// device, since it can be accessed without being inside a task
	void *_baseAddress;
	// Base (virtual) address. This is either the same as the base address when the host is the home device, or a
	// shadow region mapped in the host for the purposes of computing task dependencies. We use mmap to allocate a
	// virtual address range which is never touched, since we need some uniqueness in the addresses for dependencies
	void *_baseVirtualAddress;
	// Allocation size
	size_t _size;
	// Chunk size for each individual page
	size_t _pageSize;
	// Home device global id
	int _homeDevice;

	DirectoryPage *_pages;

public:
	DirectoryEntry(
		void *base,
		void *virtualBase,
		size_t size,
		size_t pageSize,
		int homeDeviceId,
		int maxDevices
		) :
		_baseAddress(base),
		_baseVirtualAddress(virtualBase),
		_size(size),
		_pageSize(pageSize),
		_homeDevice(homeDeviceId)
	{
		assert(size % pageSize == 0);
		char *basePtr = (char *)base;
		char *virtualBasePtr = (char *)virtualBase;
		_pages = (DirectoryPage *) MemoryAllocator::alloc(sizeof(DirectoryPage) * (size/pageSize));

		for (size_t i = 0; i < size/pageSize; ++i) {
			new (&_pages[i]) DirectoryPage(maxDevices);
			DirectoryPage &page = _pages[i];
			DirectoryPageAgentInfo &homeAgentInfo = page._agentInfo[homeDeviceId];
			homeAgentInfo._allocation = basePtr;
			homeAgentInfo._state = StateExclusive;

			// Add host region if needed
			if (homeDeviceId != 0)
				page._agentInfo[0]._allocation = virtualBasePtr;

			basePtr += pageSize;
			virtualBasePtr += pageSize;
		}
	}

	~DirectoryEntry()
	{
		for (size_t i = 0; i < getNumPages(); ++i) {
			_pages[i].~DirectoryPage();
		}

		MemoryAllocator::free(_pages, sizeof(DirectoryPage) * getNumPages());
	}

	inline size_t getPageSize() const
	{
		return _pageSize;
	}

	inline int getPageIdx(void *location) const
	{
		uintptr_t loc = (uintptr_t) location;
		uintptr_t base = (uintptr_t) _baseVirtualAddress;

		assert(base <= loc);
		assert(loc < (base + _size));

		return (loc - base) / _pageSize;
	}

	inline DirectoryPage *getPage(int idx)
	{
		return _pages + idx;
	}

	inline bool includes(void *location) const
	{
		uintptr_t locInt = (uintptr_t) location;
		uintptr_t baseInt = (uintptr_t) _baseVirtualAddress;

		return (locInt >= baseInt && locInt < (baseInt + _size));
	}

	inline size_t getSize() const
	{
		return _size;
	}

	inline void *getBaseVirtualAddress() const
	{
		return _baseVirtualAddress;
	}

	inline void *getBaseAddress() const
	{
		return _baseAddress;
	}

	inline size_t getNumPages() const
	{
		return (_size / _pageSize);
	}

	inline int getHomeDevice() const
	{
		return _homeDevice;
	}
};

#endif // DIRECTORY_ENTRY_HPP