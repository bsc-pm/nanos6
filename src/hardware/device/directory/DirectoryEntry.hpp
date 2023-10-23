/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef DIRECTORY_ENTRY_HPP
#define DIRECTORY_ENTRY_HPP

#include <cassert>

#include "DirectoryPage.hpp"

class DirectoryEntry
{
	void *_baseAddress;
	void *_baseVirtualAddress;
	size_t _size;
	size_t _pageSize;
	int _homeDevice;

	Container::vector<DirectoryPage> _pages;

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

		for (size_t i = 0; i < size/pageSize; ++i) {
			DirectoryPage &page = _pages.emplace_back(maxDevices);
			page._allocations[homeDeviceId] = basePtr;
			page._states[homeDeviceId] = StateExclusive;

			// Add host region if needed
			if (homeDeviceId != 0)
				page._allocations[0] = virtualBasePtr;

			basePtr += pageSize;
			virtualBasePtr += pageSize;
		}
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
		return &_pages[idx];
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
		return _pages.size();
	}

	inline int getHomeDevice() const
	{
		return _homeDevice;
	}
};

#endif // DIRECTORY_ENTRY_HPP