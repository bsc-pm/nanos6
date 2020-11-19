/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef MANAGER_NUMA_HPP
#define MANAGER_NUMA_HPP

#include <cstring>
#include <map>
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>

#include <nanos6.h>

#include "dependencies/DataTrackingSupport.hpp"
#include "executors/threads/CPUManager.hpp"
#include "hardware/HardwareInfo.hpp"
#include "hardware/places/NUMAPlace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "lowlevel/RWSpinLock.hpp"
#include "support/Containers.hpp"

#include <MemoryAllocator.hpp>

struct DirectoryInfo {
	size_t _size;
	uint8_t _homenode;

	DirectoryInfo(size_t size, uint8_t homenode)
		: _size(size), _homenode(homenode)
	{}
};

class ManagerNUMA {
private:
	typedef nanos6_bitmask_t bitmask_t;
	typedef Container::map<void *, DirectoryInfo> directory_t;
	typedef Container::map<void *, uint64_t> alloc_info_t;

	// Directory to store the homenode of each memory region
	static directory_t _directory;
	// RWlock to access the directory
	static RWSpinLock _lock;

	// Map to store the size of each allocation, to be able to free memory
	static alloc_info_t _allocations;
	// Lock to access the allocations map
	static SpinLock  _allocationsLock;

	// The number of CPUs that a NUMA node physically contains
	static size_t _maxCpusPerNuma;
	// The number of CPUs assigned to this process that each NUMA node contains
	static std::vector<size_t> _cpusPerNumaNode;

	// The number of NUMA nodes for wildcard NUMA_ALL
	static size_t _numNumaAll;
	// The number of NUMA nodes for wildcard NUMA_ALL_ACTIVE
	static size_t _numNumaAllActive;
	// The number of NUMA nodes for wildcard NUMA_ANY_ACTIVE
	static size_t _numNumaAnyActive;

	// The bitmask for wildcard NUMA_ALL
	static bitmask_t _bitmaskNumaAll;
	// The bitmask for wildcard NUMA_ALL_ACTIVE
	static bitmask_t _bitmaskNumaAllActive;
	// The bitmask for wildcard NUMA_ANY_ACTIVE
	static bitmask_t _bitmaskNumaAnyActive;

	static bool _reportEnabled;

#ifndef NDEBUG
	static std::atomic<size_t> _totalBytes;
	static std::atomic<size_t> _totalQueries;
#endif

public:
	static void initialize()
	{
		// Initialize bitmasks to 0
		clearAll(&_bitmaskNumaAll);
		clearAll(&_bitmaskNumaAllActive);
		clearAll(&_bitmaskNumaAnyActive);

		// Intialize _cpusPerNumaNode vector to 0
		_numNumaAll = HardwareInfo::getValidMemoryPlaceCount(nanos6_host_device);
		_cpusPerNumaNode = std::vector<size_t>(_numNumaAll, 0);

		// Get CPU list to check which CPUs we have in the process mask
		const std::vector<CPU *> &cpus = CPUManager::getCPUListReference();
		_maxCpusPerNuma = ((NUMAPlace *) HardwareInfo::getMemoryPlace(nanos6_host_device, 0))->getLocalCoreCount();

		// Iterate over the CPU list to annotate CPUs per NUMA node
		for (CPU *cpu : cpus) {
			_cpusPerNumaNode[cpu->getNumaNodeId()]++;
		}

		// Enable corresponding bits in the bitmasks
		for (size_t numaNode = 0; numaNode < _cpusPerNumaNode.size(); numaNode++) {
			// NUMA_ALL -> enable a bit per NUMA node available in the system
			enableBit(&_bitmaskNumaAll, numaNode);

			// NUMA_ANY_ACTIVE -> enable a bit per NUMA node containing at least one CPU assigned to this process
			if (_cpusPerNumaNode[numaNode] > 0) {
				enableBit(&_bitmaskNumaAnyActive, numaNode);
				_numNumaAnyActive++;

				// NUMA_ALL_ACTIVE -> enable a bit per NUMA node containing all the CPUs assigned to this process
				if (_cpusPerNumaNode[numaNode] == _maxCpusPerNuma) {
					enableBit(&_bitmaskNumaAllActive, numaNode);
					_numNumaAllActive++;
				}
			}
		}

		//_reportEnabled = true;
		if (_reportEnabled) {
			std::cout << "---------- MANAGER NUMA REPORT ----------" << std::endl;
			std::cout << "NUMA_ALL:" << std::endl;
			std::cout << "  Number of NUMA nodes: " << _numNumaAll << std::endl;
			std::cout << "  bitmask: " << _bitmaskNumaAll << std::endl;
			std::cout << "NUMA_ALL_ACTIVE:" << std::endl;
			std::cout << "  Number of NUMA nodes: " << _numNumaAllActive << std::endl;
			std::cout << "  bitmask: " << _bitmaskNumaAllActive << std::endl;
			std::cout << "NUMA_ANY_ACTIVE:" << std::endl;
			std::cout << "  Number of NUMA nodes: " << _numNumaAnyActive << std::endl;
			std::cout << "  bitmask: " << _bitmaskNumaAnyActive << std::endl;
		}

#ifndef NDEBUG
		_totalBytes = 0;
		_totalQueries = 0;
#endif
	}

	static void shutdown()
	{
#ifndef NDEBUG
		//printDirectoryContent();
#endif
		assert(_directory.empty());
	}

	static void *alloc(size_t size, bitmask_t *bitmask, size_t block_size)
	{
		assert(size > 0);

		if (!DataTrackingSupport::isNUMATrackingEnabled()) {
			void *res = malloc(size);
			FatalErrorHandler::failIf(res == nullptr, "Couldn't allocate memory.");
			return res;
		}

		assert(*bitmask != 0);
		assert(block_size > 0);

		// Check first if bitmask is any of the wildcards
		bitmask_t realBitmask;
		if (*bitmask == NUMA_ALL) {
			setAll(&realBitmask);
		} else if (*bitmask == NUMA_ALL_ACTIVE) {
			setAllActive(&realBitmask);
		} else if (*bitmask == NUMA_ANY_ACTIVE) {
			setAnyActive(&realBitmask);
		} else {
			realBitmask = *bitmask;
		}

		bitmask_t bitmaskCopy = realBitmask;
#ifndef NDEBUG
		_totalBytes += size;
#endif

		int pagesize = HardwareInfo::getPageSize();
		size_t originalBlockSize = block_size;
		if (block_size % pagesize != 0) {
			block_size = closestMultiple(block_size, pagesize);
		}

		void *res = nullptr;
		int prot = PROT_READ | PROT_WRITE;
		int flags = MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE | MAP_NONBLOCK;
		int fd = -1;
		int offset = 0;
		if (size < (size_t) pagesize) {
			// Use malloc for small allocations
			res = malloc(size);
			FatalErrorHandler::failIf(res == nullptr, "Couldn't allocate memory.");
		} else {
			// Allocate space using mmap
			void *addr = nullptr;
			res = mmap(addr, size, prot, flags, fd, offset);
			FatalErrorHandler::failIf(res == MAP_FAILED, "Couldn't allocate memory.");
		}

		_allocationsLock.lock();
		_allocations.emplace(res, size);
		_allocationsLock.unlock();

		struct bitmask *tmp_bitmask = numa_bitmask_alloc(HardwareInfo::getValidMemoryPlaceCount(nanos6_host_device));

		if (originalBlockSize < (size_t) pagesize) {
			// In this case, the whole allocation is inside the same page. However, it
			// is important for scheduling purposes to annotate in the directory as if
			// we could really split the allocation as requested.
			for (size_t i = 0; i < size; i += originalBlockSize) {
				uint8_t currentNodeIndex = indexFirstEnabledBit(bitmaskCopy);
				disableBit(&bitmaskCopy, currentNodeIndex);
				if (bitmaskCopy == 0) {
					bitmaskCopy = realBitmask;
				}

				// Insert into directory
				void *tmp = (void *) ((uintptr_t) res + i);
				size_t tmp_size = std::min(originalBlockSize, size-i);
				DirectoryInfo info(tmp_size, currentNodeIndex);
				_lock.writeLock();
				_directory.emplace(tmp, info);
				_lock.writeUnlock();
			}
		} else {
			for (size_t i = 0; i < size; i += block_size) {
				uint8_t currentNodeIndex = indexFirstEnabledBit(bitmaskCopy);
				disableBit(&bitmaskCopy, currentNodeIndex);
				if (bitmaskCopy == 0) {
					bitmaskCopy = realBitmask;
				}

				// Place pages where they must be
				void *tmp = (void *) ((uintptr_t) res + i);
				numa_bitmask_clearall(tmp_bitmask);
				numa_bitmask_setbit(tmp_bitmask, currentNodeIndex);
				assert(numa_bitmask_isbitset(tmp_bitmask, currentNodeIndex));
				size_t tmp_size = std::min(block_size, size-i);
				numa_interleave_memory(tmp, tmp_size, tmp_bitmask);

				// Insert into directory
				DirectoryInfo info(tmp_size, currentNodeIndex);
				_lock.writeLock();
				_directory.emplace(tmp, info);
				_lock.writeUnlock();
			}
		}

		numa_bitmask_free(tmp_bitmask);

#ifndef NDEBUG
		int pid = 0;
		unsigned long numPages = ceil(size, pagesize);
		void **pages = (void **) MemoryAllocator::alloc(numPages * sizeof(void *));
		int *nodes = nullptr;
		int *status = (int *) MemoryAllocator::alloc(numPages * sizeof(int));
		flags = 0;
		size_t page = 0;
		uint8_t currentNodeIndex = indexFirstEnabledBit(bitmaskCopy);
		disableBit(&bitmaskCopy, currentNodeIndex);
		if (bitmaskCopy == 0) {
			bitmaskCopy = realBitmask;
		}

		size_t blockBytes = 0;
		for (size_t i = 0; i < size; i += pagesize) {
			if (blockBytes >= block_size) {
				currentNodeIndex = indexFirstEnabledBit(bitmaskCopy);
				disableBit(&bitmaskCopy, currentNodeIndex);
				if (bitmaskCopy == 0) {
					bitmaskCopy = realBitmask;
				}
				blockBytes = 0;
			}

			char *tmp = (char *) res+i;
			tmp[0] = 0;
			pages[page] = tmp;

			blockBytes += pagesize;
			page++;
		}
		assert(numPages == page);

		// move_pages
		__attribute__((unused)) long ret = move_pages(pid, numPages, pages, nodes, status, flags);
		assert(ret == 0);

		// Check pages are properly distributed
		bitmaskCopy = realBitmask;
		currentNodeIndex = indexFirstEnabledBit(bitmaskCopy);
		disableBit(&bitmaskCopy, currentNodeIndex);
		if (bitmaskCopy == 0) {
			bitmaskCopy = realBitmask;
		}

		blockBytes = 0;
		for (size_t i = 0; i < numPages; i++) {
			if (blockBytes >= block_size) {
				currentNodeIndex = indexFirstEnabledBit(bitmaskCopy);
				disableBit(&bitmaskCopy, currentNodeIndex);
				if (bitmaskCopy == 0) {
					bitmaskCopy = realBitmask;
				}
				blockBytes = 0;
			}

			assert(status[i] >= 0);
			FatalErrorHandler::warnIf(status[i] != currentNodeIndex, "Page is not where it should.");

			blockBytes += pagesize;
		}

		MemoryAllocator::free(pages, numPages * sizeof(void *));
		MemoryAllocator::free(status, numPages * sizeof(int));

#endif

		return res;
	}

	static void freeDebug(void *ptr, size_t size, bitmask_t *bitmask, size_t block_size)
	{
		assert(size > 0);
		assert(*bitmask > 0);
		assert(block_size > 0);
		size_t originalBlockSize = block_size;
		int pagesize = HardwareInfo::getPageSize();

		if (DataTrackingSupport::isNUMATrackingEnabled()) {
			bitmask_t bitmaskCopy = *bitmask;

			if (block_size % pagesize != 0) {
				block_size = closestMultiple(block_size, pagesize);
				//FatalErrorHandler::warnIf(true, "Block size is not multiple of pagesize. Using ", block_size, " instead.");
			}

#ifndef NDEBUG
			size_t remainingBytes = size;
#endif
			if (originalBlockSize < (size_t) pagesize) {
				for (size_t i = 0; i < size; i += originalBlockSize) {
					uint8_t currentNodeIndex = indexFirstEnabledBit(bitmaskCopy);
					disableBit(&bitmaskCopy, currentNodeIndex);
					if (bitmaskCopy == 0) {
						bitmaskCopy = *bitmask;
					}

					// Place pages where they must be
					void *tmp = (void *) ((uintptr_t) ptr + i);
#ifndef NDEBUG
					_lock.readLock();
					auto it = _directory.find(tmp);
					assert(it->second._size == (size_t) originalBlockSize || it->second._size == remainingBytes);
					_lock.readUnlock();
					remainingBytes -= it->second._size;
#endif
					_lock.writeLock();
					__attribute__((unused)) size_t numErased = _directory.erase(tmp);
					_lock.writeUnlock();
					assert(numErased == 1);
				}
			} else {
				for (size_t i = 0; i < size; i += block_size) {
					uint8_t currentNodeIndex = indexFirstEnabledBit(bitmaskCopy);
					disableBit(&bitmaskCopy, currentNodeIndex);
					if (bitmaskCopy == 0) {
						bitmaskCopy = *bitmask;
					}

					// Place pages where they must be
					void *tmp = (void *) ((uintptr_t) ptr + i);
#ifndef NDEBUG
					_lock.readLock();
					auto it = _directory.find(tmp);
					assert(it->second._size == block_size || it->second._size == remainingBytes);
					_lock.readUnlock();
					remainingBytes -= it->second._size;
#endif
					_lock.writeLock();
					__attribute__((unused)) size_t numErased = _directory.erase(tmp);
					_lock.writeUnlock();
					assert(numErased == 1);
				}
			}

			// Release memory
			if (size < (size_t) pagesize) {
				std::free(ptr);
			} else {
				__attribute__((unused)) int res = munmap(ptr, size);
				assert(res == 0);
			}
		} else {
			std::free(ptr);
		}
	}

	static void free(void *ptr)
	{
		int pagesize = HardwareInfo::getPageSize();

		_allocationsLock.lock();
		size_t size = _allocations[ptr];
		_allocationsLock.unlock();

		void *aux = (void *)ptr;
		do {
			_lock.readLock();
			auto it = _directory.find(aux);
			_lock.readUnlock();

			if (it == _directory.end()) {
				break;
			}

			void *toErase = aux;
			it++;
			aux = it->first;

			_directory.erase(toErase);
		} while (aux < (char *)(ptr)+size);

		// Release memory
		if (size < (size_t) pagesize) {
			std::free(ptr);
		} else {
			__attribute__((unused)) int res = munmap(ptr, size);
			assert(res == 0);
		}
	}

	static uint8_t getHomeNode(void *ptr, size_t size)
	{
		if (!DataTrackingSupport::isNUMATrackingEnabled()) {
			return (uint8_t) -1;
		}

#ifndef NDEBUG
		_totalQueries++;
#endif

		uint8_t homenode = (uint8_t ) -1;
		// Search in the directory
		_lock.readLock();
		auto it = _directory.lower_bound(ptr);
		_lock.readUnlock();

		// lower_bound returns the first element not considered to go before ptr
		// Thus, if ptr is exactly the start of the region, lower_bound will return
		// the desired region. Otherwise, if ptr belongs to the region but its start
		// address is greater than the region start, lower_bound returns the next region.
		// In consequence, we should apply a decrement to the it.
		// Get homenode
		if (it == _directory.end() || ptr < it->first) {
			if (it == _directory.begin()) {
				return homenode;
			}
			it--;
		}

		// Not present
		if (it == _directory.end()) {
			return homenode;
		}

		// It could happen that the region we are checking resides in several directory regions.
		// We will return as the homenode the one containing more bytes.
		size_t foundBytes = 0;
		size_t *bytesInNUMA = (size_t *) alloca(HardwareInfo::getValidMemoryPlaceCount(nanos6_host_device)*sizeof(size_t));
		std::memset(bytesInNUMA, 0, HardwareInfo::getValidMemoryPlaceCount(nanos6_host_device)*sizeof(size_t));
		int idMax = -1;
		do {
			size_t containedBytes = getContainedBytes((uintptr_t) it->first, it->second._size, (uintptr_t) ptr, size);
			homenode = it->second._homenode;
			bytesInNUMA[homenode] += containedBytes;

			if (idMax == -1 || bytesInNUMA[homenode] > bytesInNUMA[idMax]) {
				idMax = homenode;
			}

			// Cutoff
			if (bytesInNUMA[homenode] >= (size/2)) {
				assert(homenode != (uint8_t) -1);
				return homenode;
			}

			foundBytes += containedBytes;
			it++;
		} while (foundBytes != size && it != _directory.end());
		assert(idMax != -1);

		return idMax;
	}

	static inline void clearAll(bitmask_t *bitmask)
	{
		*bitmask = 0;
	}

	static inline void clearBit(bitmask_t *bitmask, uint64_t bitIndex)
	{
		disableBit(bitmask, bitIndex);
	}

	static inline void setAll(bitmask_t *bitmask)
	{
		*bitmask = _bitmaskNumaAll;
	}

	static inline void setAllActive(bitmask_t *bitmask)
	{
		*bitmask = _bitmaskNumaAllActive;
	}

	static inline void setAnyActive(bitmask_t *bitmask)
	{
		*bitmask = _bitmaskNumaAnyActive;
	}

	static inline void setBit(bitmask_t *bitmask, uint64_t bitIndex)
	{
		enableBit(bitmask, bitIndex);
	}

	static inline uint8_t isBitSet(bitmask_t *bitmask, uint64_t bitIndex)
	{
		return checkBit(bitmask, bitIndex);
	}

	static inline uint8_t getNumaNodes(bitmask_t *bitmask)
	{
		// In this method, we can only use the three wildcards described in numa.h:
		//	 - NUMA_ALL: all the NUMA nodes available in the system
		//	 - NUMA_ALL_ACTIVE: the NUMA nodes where we have all the CPUs assigned
		//	 - NUMA_ANY_ACTIVE: the NUMA nodes where we have any of the CPUs assigned
		// Any other value is not accepted.
		if (*bitmask == NUMA_ALL) {
			return _numNumaAll;
		} else if (*bitmask == NUMA_ALL_ACTIVE) {
			return _numNumaAllActive;
		} else if (*bitmask == NUMA_ANY_ACTIVE) {
			return _numNumaAnyActive;
		} else {
			FatalErrorHandler::warnIf(true, "Unknown bitmask value. Defaulting to NUMA_ALL.");
			return _numNumaAll;
		}
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

	static inline void enableBit(uint64_t *x, uint64_t bitIndex)
	{
		*x |= ((uint64_t) 1 << bitIndex);
	}

	static inline uint8_t checkBit(uint64_t *x, uint64_t bitIndex)
	{
		uint8_t bit = ((*x >> bitIndex) & (uint64_t) 1);
		return bit;
	}

	static inline size_t getContainedBytes(uintptr_t ptr1, size_t size1, uintptr_t ptr2, size_t size2)
	{
		uintptr_t end1 = (uintptr_t) ptr1 + size1;
		uintptr_t end2 = (uintptr_t) ptr2 + size2;

		if (end1 <= ptr2)
			return 0;

		if (end2 <= ptr1)
			return 0;

		if (ptr1 > ptr2) {
			return end2 - ptr1;
		} else {
			if (end2 > end1)
				return end1 - ptr2;
			else
				return end2 - ptr2;
		}
	}

	static void printDirectoryContent()
	{
		if (_directory.empty()) {
			std::cout << "Directory is empty." << std::endl;
			return;
		}
		for (auto it : _directory) {
			std::cout << "Address: " << it.first << ", size: " << it.second._size << ", homenode: " << (int) it.second._homenode << std::endl;
		}
	}
};

#endif //MANAGER_NUMA_HPP
