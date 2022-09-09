/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef MANAGER_NUMA_HPP
#define MANAGER_NUMA_HPP

#include <cstring>
#include <map>
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>

#include <nanos6.h>

#include "executors/threads/CPUManager.hpp"
#include "hardware/HardwareInfo.hpp"
#include "hardware/places/NUMAPlace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "lowlevel/RWSpinLock.hpp"
#include "support/BitManipulation.hpp"
#include "support/Containers.hpp"
#include "support/MathSupport.hpp"
#include "support/config/ConfigVariable.hpp"

#include <DataAccessRegistration.hpp>
#include <MemoryAllocator.hpp>

struct DirectoryInfo {
	size_t _size;
	uint8_t _homeNode;

	DirectoryInfo(size_t size, uint8_t homeNode) :
		_size(size),
		_homeNode(homeNode)
	{
	}
};

class NUMAManager {
private:
	typedef nanos6_bitmask_t bitmask_t;
	typedef Container::map<void *, DirectoryInfo> directory_t;
	typedef Container::map<void *, uint64_t> alloc_info_t;

	//! Directory to store the homeNode of each memory region
	static directory_t _directory;

	//! RWlock to access the directory
	static RWSpinLock _lock;

	//! Map to store the size of each allocation, to be able to free memory
	static alloc_info_t _allocations;

	//! Lock to access the allocations map
	static SpinLock _allocationsLock;

	//! The bitmask for wildcard NUMA_ALL
	static bitmask_t _bitmaskNumaAll;

	//! The bitmask for wildcard NUMA_ALL_ACTIVE
	static bitmask_t _bitmaskNumaAllActive;

	//! The bitmask for wildcard NUMA_ANY_ACTIVE
	static bitmask_t _bitmaskNumaAnyActive;

	//! Whether the tracking is enabled
	static std::atomic<bool> _trackingEnabled;

	//! The tracking mode "on", "off" or "auto"
	static ConfigVariable<std::string> _trackingMode;

	//! Whether should report NUMA information
	static ConfigVariable<bool> _reportEnabled;

	//! Wether the automatic page discovery is enabled or disabled
	static ConfigVariable<bool> _discoverPageSize;

	//! Whether the real pagesize must be discovered
	static bool _mustDiscoverRealPageSize;

	//! Maximum OS Index for a NUMA node
	static int _maxOSIndex;

	//! Array to get the corresponding OS index of a logical index
	static std::vector<int> _logicalToOsIndex;

public:
	static void initialize()
	{
		_trackingEnabled = false;
		std::string trackingMode = _trackingMode.getValue();
		if (trackingMode == "on") {
			// Mark tracking as enabled
			_trackingEnabled = true;
		} else if (trackingMode != "auto" && trackingMode != "off") {
			FatalErrorHandler::fail("Invalid data tracking mode: ", trackingMode);
		}

		// We always initialize everything, even in the "off" case. If "auto" we
		// enable the tracking in the first alloc/allocSentinels call

		// Initialize bitmasks to zero
		clearAll(&_bitmaskNumaAll);
		clearAll(&_bitmaskNumaAllActive);
		clearAll(&_bitmaskNumaAnyActive);

		size_t numNumaAll = 0;
		size_t numNumaAllActive = 0;
		size_t numNumaAnyActive = 0;

		numNumaAll = HardwareInfo::getMemoryPlaceCount(nanos6_host_device);
		_logicalToOsIndex.resize(numNumaAll);

		// Currently, we are using uint64_t as type for the bitmasks. In case we have
		// more than 64 nodes, the bitmask cannot represent all the NUMA nodes
		FatalErrorHandler::failIf((numNumaAll > 64), "We cannot support such a high number of NUMA nodes.");
		FatalErrorHandler::failIf((numNumaAll <= 0), "There must be at least one NUMA node.");

		// The number of CPUs assigned to this process that each NUMA node contains
		std::vector<size_t> cpusPerNumaNode(numNumaAll, 0);

		// Get CPU list to check which CPUs we have in the process mask
		const std::vector<CPU *> &cpus = CPUManager::getCPUListReference();
		for (CPU *cpu : cpus) {
			assert(cpu != nullptr);

			size_t numaId = cpu->getNumaNodeId();
			assert(numaId < numNumaAll);

			// If DLB is enabled, we only want the CPUs we own
			if (cpu->isOwned()) {
				cpusPerNumaNode[numaId]++;
			}
		}

		_maxOSIndex = -1;

		// Enable corresponding bits in the bitmasks
		for (size_t numaNode = 0; numaNode < cpusPerNumaNode.size(); numaNode++) {
			NUMAPlace *numaPlace = (NUMAPlace *) HardwareInfo::getMemoryPlace(nanos6_host_device, numaNode);
			assert(numaPlace != nullptr);

			// As we will interact with libnuma, we need to use the OS index in the bitmask
			int osIndex = numaPlace->getOsIndex();
			_logicalToOsIndex[numaNode] = osIndex;
			if (osIndex != -1) {
				// NUMA_ALL enables a bit per NUMA node available in the system
				BitManipulation::enableBit(&_bitmaskNumaAll, numaNode);

				// NUMA_ANY_ACTIVE enables a bit per NUMA node containing at least one CPU assigned to this process
				if (cpusPerNumaNode[numaNode] > 0) {
					BitManipulation::enableBit(&_bitmaskNumaAnyActive, numaNode);
					numNumaAnyActive++;

					// NUMA_ALL_ACTIVE enables a bit per NUMA node containing all the CPUs assigned to this process
					if (cpusPerNumaNode[numaNode] == numaPlace->getNumLocalCores()) {
						BitManipulation::enableBit(&_bitmaskNumaAllActive, numaNode);
						numNumaAllActive++;
					}
				}
			}

			if (osIndex > _maxOSIndex) {
				_maxOSIndex = osIndex;
			}
		}
		assert(_maxOSIndex != -1);

		// Page auto-discovery will be enabled if we have at least two active NUMA nodes and tracking is enabled or automatic
		if (_discoverPageSize.getValue() && (trackingMode == "auto" || trackingMode == "on") && numNumaAnyActive > 1) {
			_mustDiscoverRealPageSize = true;
		} else {
			_mustDiscoverRealPageSize = false;
		}

		if (_reportEnabled) {
			FatalErrorHandler::print("---------- MANAGER NUMA REPORT ----------");
			FatalErrorHandler::print("NUMA_ALL:");
			FatalErrorHandler::print("  Number of NUMA nodes: ", numNumaAll);
			FatalErrorHandler::print("  bitmask: ", _bitmaskNumaAll);
			FatalErrorHandler::print("NUMA_ALL_ACTIVE:");
			FatalErrorHandler::print("  Number of NUMA nodes: ", numNumaAllActive);
			FatalErrorHandler::print("  bitmask: ", _bitmaskNumaAllActive);
			FatalErrorHandler::print("NUMA_ANY_ACTIVE:");
			FatalErrorHandler::print("  Number of NUMA nodes: ", numNumaAnyActive);
			FatalErrorHandler::print("  bitmask: ", _bitmaskNumaAnyActive);
		}
	}

	static void shutdown()
	{
		assert(_directory.empty());
		assert(_allocations.empty());
	}

	static void *alloc(size_t size, const bitmask_t *bitmask, size_t blockSize)
	{
		size_t pageSize = HardwareInfo::getPageSize();
		assert(pageSize > 0);

		if (!enableTrackingIfAuto()) {
			void *res = nullptr;
			if (size < pageSize) {
				res = malloc(size);
			} else {
				int err = posix_memalign(&res, pageSize, size);
				FatalErrorHandler::failIf(err != 0);
			}
			FatalErrorHandler::failIf(res == nullptr, "Couldn't allocate memory.");
			return res;
		}

		size_t realPageSize = getRealPageSize();
		assert(realPageSize != 0);

		// To explain the following code, let us assume a huge page is 2MB, and
		// a normal system page is 4KB:
		// - If the allocation size is < 4KB, we use malloc
		// - If the allocation size is > 4KB, we use mmap and inertwine memory
		//   between NUMA nodes using "blockSize" in each
		if (size < pageSize) {
			void *res = malloc(size);
			FatalErrorHandler::failIf(res == nullptr, "Couldn't allocate memory.");
			return res;
		}

		assert(bitmask != nullptr);
		assert(*bitmask != 0);
		assert(blockSize > 0);

		// If we're allocating more than THP size, use that as page size
		if (size > realPageSize) {
			pageSize = realPageSize;
		}
		bitmask_t bitmaskCopy = *bitmask;
		if (blockSize % pageSize != 0) {
			blockSize = MathSupport::closestMultiple(blockSize, pageSize);
		}

		void *res = nullptr;
		{
			// Allocate space using mmap
			int prot = PROT_READ | PROT_WRITE;
			int flags = MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE | MAP_NONBLOCK;
			int fd = -1;
			int offset = 0;
			void *addr = nullptr;
			res = mmap(addr, size, prot, flags, fd, offset);
			FatalErrorHandler::failIf(res == MAP_FAILED, "Couldn't allocate memory.");
		}

		_allocationsLock.lock();
		_allocations.emplace(res, size);
		_allocationsLock.unlock();

		struct bitmask *tmpBitmask = numa_bitmask_alloc(_maxOSIndex + 1);
		for (size_t i = 0; i < size; i += blockSize) {
			uint8_t currentNodeIndex = BitManipulation::indexFirstEnabledBit(bitmaskCopy);
			BitManipulation::disableBit(&bitmaskCopy, currentNodeIndex);
			if (bitmaskCopy == 0) {
				bitmaskCopy = *bitmask;
			}

			void *tmp = (void *) ((uintptr_t) res + i);
			size_t tmpSize = std::min(blockSize, size-i);

			// Set all the pages of a block in the same node.
			numa_bitmask_clearall(tmpBitmask);
			assert(_logicalToOsIndex[currentNodeIndex] != -1);

			numa_bitmask_setbit(tmpBitmask, _logicalToOsIndex[currentNodeIndex]);
			assert(numa_bitmask_isbitset(tmpBitmask, _logicalToOsIndex[currentNodeIndex]));

			numa_interleave_memory(tmp, tmpSize, tmpBitmask);

			// Insert into directory
			DirectoryInfo info(tmpSize, currentNodeIndex);
			_lock.writeLock();
			_directory.emplace(tmp, info);
			_lock.writeUnlock();
		}
		numa_bitmask_free(tmpBitmask);

#ifndef NDEBUG
		checkAllocationCorrectness(res, size, bitmask, blockSize);
#endif

		return res;
	}

	static void *allocSentinels(size_t size, const bitmask_t *bitmask, size_t blockSize)
	{
		assert(size > 0);

		size_t pageSize = HardwareInfo::getPageSize();
		if (!enableTrackingIfAuto()) {
			void *res = nullptr;
			if (size < pageSize) {
				res = malloc(size);
			} else {
				int err = posix_memalign(&res, pageSize, size);
				FatalErrorHandler::failIf(err != 0);
			}
			FatalErrorHandler::failIf(res == nullptr, "Couldn't allocate memory.");
			return res;
		}

		assert(*bitmask != 0);
		assert(blockSize > 0);

		bitmask_t bitmaskCopy = *bitmask;
		size_t realPageSize = getRealPageSize();
		assert(realPageSize != 0);

		pageSize = (size <= realPageSize) ? pageSize : realPageSize;
		assert(pageSize > 0);

		void *res = nullptr;
		if (size < pageSize) {
			// Use malloc for small allocations
			res = malloc(size);
			FatalErrorHandler::failIf(res == nullptr, "Couldn't allocate memory.");
		} else {
			// Allocate space using mmap
			int prot = PROT_READ | PROT_WRITE;
			int flags = MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE | MAP_NONBLOCK;
			int fd = -1;
			int offset = 0;
			void *addr = nullptr;
			res = mmap(addr, size, prot, flags, fd, offset);
			FatalErrorHandler::failIf(res == MAP_FAILED, "Couldn't allocate memory.");
		}

		_allocationsLock.lock();
		_allocations.emplace(res, size);
		_allocationsLock.unlock();

		// In this case, the whole allocation is inside the same page. However, it
		// is important for scheduling purposes to annotate in the directory as if
		// we could really split the allocation as requested
		for (size_t i = 0; i < size; i += blockSize) {
			uint8_t currentNodeIndex = BitManipulation::indexFirstEnabledBit(bitmaskCopy);
			BitManipulation::disableBit(&bitmaskCopy, currentNodeIndex);
			if (bitmaskCopy == 0) {
				bitmaskCopy = *bitmask;
			}

			// Insert into directory
			void *tmp = (void *) ((uintptr_t) res + i);
			size_t tmpSize = std::min(blockSize, size - i);
			DirectoryInfo info(tmpSize, currentNodeIndex);
			_lock.writeLock();
			_directory.emplace(tmp, info);
			_lock.writeUnlock();
		}


		return res;
	}

	static void free(void *ptr)
	{
		if (!isTrackingEnabled()) {
			std::free(ptr);
			return;
		}

		_allocationsLock.lock();
		// Find the allocation size and remove (one single map search)
		auto allocIt = _allocations.find(ptr);

		// In some cases, in the alloc methods, we simply use the standard
		// malloc and we do not annotate that in the map. Thus, simply
		// release the lock and use standard free.
		if (allocIt == _allocations.end()) {
			_allocationsLock.unlock();
			std::free(ptr);
			return;
		}

		size_t size = allocIt->second;
		_allocations.erase(allocIt);
		_allocationsLock.unlock();

		_lock.writeLock();
		// Find the initial element in the directory
		auto begin = _directory.find(ptr);
		assert(begin != _directory.end());

		// Find the next element after the allocation
		auto end = _directory.lower_bound((void *) ((uintptr_t) ptr + size));

		// Remove all elements in the range [begin, end)
		_directory.erase(begin, end);
		_lock.writeUnlock();

		// Release memory
		size_t pageSize = HardwareInfo::getPageSize();
		size_t realPageSize = getRealPageSize();
		pageSize = (size <= realPageSize) ? pageSize : realPageSize;
		if (size < pageSize) {
			std::free(ptr);
		} else {
			__attribute__((unused)) int res = munmap(ptr, size);
			assert(res == 0);
		}
	}

	static inline uint8_t getHomeNode(void *ptr, size_t size)
	{
		if (!isTrackingEnabled()) {
			return (uint8_t) -1;
		} else {
			return doGetHomeNode(ptr, size);
		}
	}

	static inline void clearAll(bitmask_t *bitmask)
	{
		*bitmask = 0;
	}

	static inline void clearBit(bitmask_t *bitmask, uint64_t bitIndex)
	{
		BitManipulation::disableBit(bitmask, bitIndex);
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

	static inline void setWildcard(bitmask_t *bitmask, nanos6_bitmask_wildcard_t wildcard)
	{
		if (wildcard == NUMA_ALL) {
			setAll(bitmask);
		} else if (wildcard == NUMA_ALL_ACTIVE) {
			setAllActive(bitmask);
		} else if (wildcard == NUMA_ANY_ACTIVE) {
			setAnyActive(bitmask);
		} else {
			FatalErrorHandler::warnIf(true, "No valid wildcard provided. Bitmask is left unchangend.");
		}
	}

	static inline void setBit(bitmask_t *bitmask, uint64_t bitIndex)
	{
		BitManipulation::enableBit(bitmask, bitIndex);
	}

	static inline uint64_t isBitSet(const bitmask_t *bitmask, uint64_t bitIndex)
	{
		return (uint64_t) BitManipulation::checkBit(bitmask, bitIndex);
	}

	static inline uint64_t countEnabledBits(const bitmask_t *bitmask)
	{
		return BitManipulation::countEnabledBits(bitmask);
	}

	static inline bool isTrackingEnabled()
	{
		return (_trackingEnabled.load(std::memory_order_relaxed) &&
			getValidTrackingNodes() > 1 &&
			DataAccessRegistration::supportsDataTracking());
	}

	static inline bool isValidNUMA(uint64_t bitIndex)
	{
		return BitManipulation::checkBit(&_bitmaskNumaAnyActive, bitIndex);
	}

	static inline size_t getOSIndex(size_t logicalId)
	{
		assert(logicalId < _logicalToOsIndex.size());
		return _logicalToOsIndex[logicalId];
	}

	static uint64_t getTrackingNodes();

private:
	static inline uint8_t doGetHomeNode(void *ptr, size_t size)
	{
		// Search in the directory
		_lock.readLock();
		auto it = _directory.lower_bound(ptr);

		// lower_bound returns the first element not considered to go before ptr
		// Thus, if ptr is exactly the start of the region, lower_bound will return
		// the desired region. Otherwise, if ptr belongs to the region but its start
		// address is greater than the region start, lower_bound returns the next
		// region. In consequence, we should apply a decrement to the iterator
		if (it == _directory.end() || ptr < it->first) {
			if (it == _directory.begin()) {
				_lock.readUnlock();
				return (uint8_t) -1;
			}
			it--;
		}

		// Not present
		if (it == _directory.end() || getContainedBytes(ptr, size, it->first, it->second._size) == 0) {
			_lock.readUnlock();
			return (uint8_t) -1;
		}

		// If the target region resides in several directory regions, we return as the
		// homeNode the one containing more bytes

		size_t numNumaAll = HardwareInfo::getMemoryPlaceCount(nanos6_host_device);
		assert(numNumaAll > 0);

		size_t *bytesInNUMA = (size_t *) alloca(numNumaAll * sizeof(size_t));
		std::memset(bytesInNUMA, 0, numNumaAll * sizeof(size_t));

		int idMax = 0;
		size_t foundBytes = 0;
		do {
			size_t containedBytes = getContainedBytes(it->first, it->second._size, ptr, size);

			// Break after we are out of the range [ptr, end)
			if (containedBytes == 0)
				break;

			uint8_t homeNode = it->second._homeNode;
			assert(homeNode != (uint8_t) -1);
			bytesInNUMA[homeNode] += containedBytes;

			if (bytesInNUMA[homeNode] > bytesInNUMA[idMax]) {
				idMax = homeNode;
			}

			// Cutoff: no other NUMA node can score better than this
			if (bytesInNUMA[homeNode] >= (size / 2)) {
				_lock.readUnlock();
				return homeNode;
			}

			foundBytes += containedBytes;
			it++;
		} while (foundBytes != size && it != _directory.end());
		_lock.readUnlock();

		assert(bytesInNUMA[idMax] > 0);

		return idMax;
	}

	static inline size_t getContainedBytes(void *ptr1, size_t size1, void *ptr2, size_t size2)
	{
		uintptr_t start1 = (uintptr_t) ptr1;
		uintptr_t start2 = (uintptr_t) ptr2;
		uintptr_t end1 = start1 + size1;
		uintptr_t end2 = start2 + size2;
		uintptr_t start = std::max(start1, start2);
		uintptr_t end = std::min(end1, end2);

		if (start < end)
			return end - start;

		return 0;
	}

	static bool enableTrackingIfAuto()
	{
		if (isTrackingEnabled()) {
			return true;
		}

		std::string trackingMode = _trackingMode.getValue();
		if (trackingMode == "auto" && getValidTrackingNodes() > 1 && DataAccessRegistration::supportsDataTracking()) {
			_trackingEnabled.store(true, std::memory_order_release);
			return true;
		}

		return false;
	}

	static inline uint64_t getValidTrackingNodes()
	{
		std::string trackingMode = _trackingMode.getValue();
		if (trackingMode == "off") {
			return 1;
		} else {
			return BitManipulation::countEnabledBits(&_bitmaskNumaAnyActive);
		}
	}

	static size_t getRealPageSize();

	static size_t discoverRealPageSize();

#ifndef NDEBUG
	static void checkAllocationCorrectness(
		void *res, size_t size,
		const bitmask_t *bitmask,
		size_t blockSize
	);
#endif
};

#endif //MANAGER_NUMA_HPP
