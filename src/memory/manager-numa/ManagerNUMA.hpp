/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef MANAGER_NUMA_HPP
#define MANAGER_NUMA_HPP

#include <map>
#include <numaif.h>

#include <nanos6.h>

#include <MemoryAllocator.hpp>

#include "hardware/HardwareInfo.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "lowlevel/RWSpinLock.hpp"

class ManagerNUMA {
private:
	typedef nanos6_bitmask_t bitmask_t;
	// All what is stored here are pages. So the size is always pagesize.
	// Therefore, we only need the initial address and the homenode.
	typedef std::map<void *, uint8_t> directory_t;

	static directory_t _directory;
	static RWSpinLock _lock;

public:
	static void initialize()
	{}

	static void shutdown()
	{
		assert(_directory.empty());
	}

	static void *alloc(size_t size, bitmask_t *bitmask, size_t block_size)
	{
		assert(size > 0);
		assert(*bitmask > 0);
		assert(block_size > 0);

		bitmask_t bitmaskCopy = *bitmask;

		int pagesize = HardwareInfo::getPageSize();
		if (block_size % pagesize != 0) {
			block_size = closestMultiple(block_size, pagesize);
			FatalErrorHandler::warnIf(true, "Block size is not multiple of pagesize. Using ", block_size, " instead.");
		}

		// PID = 0 means move pages of this process.
		int pid = 0;
		unsigned long numPages = ceil(size, pagesize);
		void **pages = (void **) MemoryAllocator::alloc(numPages * sizeof(void *));
		int *nodes = (int *) MemoryAllocator::alloc(numPages * sizeof(int));
		int *status = (int *) MemoryAllocator::alloc(numPages * sizeof(int));
		int flags = 0;
		size_t page = 0;
		uint8_t currentNodeIndex = indexFirstEnabledBit(bitmaskCopy);
		disableBit(&bitmaskCopy, currentNodeIndex);
		if (bitmaskCopy == 0) {
			bitmaskCopy = *bitmask;
		}

		// Allocate memory
		void *res = aligned_alloc(pagesize, size);
		assert(res != nullptr);

		// First touch, otherwise move_pages will fail
		size_t blockBytes = 0;
		for (size_t i = 0; i < size; i += pagesize) {
			if (blockBytes >= block_size) {
				currentNodeIndex = indexFirstEnabledBit(bitmaskCopy);
				disableBit(&bitmaskCopy, currentNodeIndex);
				if (bitmaskCopy == 0) {
					bitmaskCopy = *bitmask;
				}
				blockBytes = 0;
			}

			char *tmp = (char *) res+i;
			tmp[0] = 0;
			pages[page] = tmp;
			nodes[page] = currentNodeIndex;

			_lock.writeLock();
			_directory.emplace(tmp, currentNodeIndex);
			_lock.writeUnlock();

			blockBytes += pagesize;
			page++;
		}
		assert(numPages == page);

		// move_pages
		__attribute__((unused)) long ret = move_pages(pid, numPages, pages, nodes, status, flags);
		assert(ret == 0);

		// Check pages are properly distributed
		bitmaskCopy = *bitmask;
		currentNodeIndex = indexFirstEnabledBit(bitmaskCopy);
		disableBit(&bitmaskCopy, currentNodeIndex);
		if (bitmaskCopy == 0) {
			bitmaskCopy = *bitmask;
		}

#ifndef NDEBUG
		blockBytes = 0;
		for (size_t i = 0; i < numPages; i++) {
			if (blockBytes >= block_size) {
				currentNodeIndex = indexFirstEnabledBit(bitmaskCopy);
				disableBit(&bitmaskCopy, currentNodeIndex);
				if (bitmaskCopy == 0) {
					bitmaskCopy = *bitmask;
				}
				blockBytes = 0;
			}

			assert(status[i] >= 0);
			assert(status[i] == currentNodeIndex);

			blockBytes += pagesize;
		}
#endif

		MemoryAllocator::free(pages, numPages * sizeof(void *));
		MemoryAllocator::free(nodes, numPages * sizeof(int));
		MemoryAllocator::free(status, numPages * sizeof(int));

		return res;
	}

	static void free(void *ptr, size_t size)
	{
		int pagesize = HardwareInfo::getPageSize();
		assert((size_t) ptr % pagesize == 0);

		// Remove all pages from directory
		for (size_t i = 0; i < size; i += pagesize) {
			char *tmp = (char *) ptr+i;

			_lock.writeLock();
			__attribute__((unused)) size_t numErased = _directory.erase(tmp);
			_lock.writeUnlock();
			assert(numErased == 1);
		}

		// Release memory
		std::free(ptr);
	}

	static uint8_t getHomeNode(void *ptr)
	{
		// Align to pagesize
		void *alignedPtr = (void *) closestMultiple((size_t) ptr, HardwareInfo::getPageSize());

		// Search in the directory
		_lock.readLock();
		auto it = _directory.find(alignedPtr);
		_lock.readUnlock();

		// Not present
		if (it == _directory.end())
			return (uint8_t) -1;

		// Get homenode
		uint8_t homenode = it->second;
		assert(homenode != (uint8_t) -1);

		return homenode;
	}

private:
	static inline size_t ceil(size_t x, size_t y)
	{
		return (x+(y-1))/y;
	}

	static inline size_t closestMultiple(size_t n, size_t multipleOf)
	{
		return ((n + multipleOf - 1) / multipleOf) * multipleOf;
	}

	// ffs returns the least signficant enabled bit, starting from 1
	// 0 means x has no enabled bits
	static inline int indexFirstEnabledBit(uint64_t x)
	{
		return __builtin_ffsll(x) - 1;
	}

	static inline void disableBit(uint64_t *x, uint64_t bitIndex)
	{
		*x &= ~((uint64_t) 1 << bitIndex);
	}
};

#endif //MANAGER_NUMA_HPP
