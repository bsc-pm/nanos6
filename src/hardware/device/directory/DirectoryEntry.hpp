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
	size_t _size;
	size_t _pageSize;

	Container::vector<DirectoryPage> _pages;

public:
	DirectoryEntry(
		void *base,
		size_t size,
		size_t pageSize,
		int homeDeviceId,
		int maxDevices
		) :
		_baseAddress(base),
		_size(size),
		_pageSize(pageSize)
	{
		assert(size % pageSize == 0);
		char *basePtr = (char *)base;

		for (size_t i = 0; i < size/pageSize; ++i) {
			DirectoryPage &page = _pages.emplace_back(maxDevices);
			page._homeDevice = homeDeviceId;
			page._allocations[homeDeviceId] = basePtr;
			page._states[homeDeviceId] = StateExclusive;
			basePtr += pageSize;
		}
	}

	inline size_t getPageSize() const
	{
		return _pageSize;
	}

	inline int getPageIdx(void *location) const
	{
		uintptr_t loc = (uintptr_t) location;
		uintptr_t base = (uintptr_t) _baseAddress;

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
		uintptr_t baseInt = (uintptr_t) _baseAddress;

		return (locInt >= baseInt && locInt < (baseInt + _size));
	}
};

#endif // DIRECTORY_ENTRY_HPP