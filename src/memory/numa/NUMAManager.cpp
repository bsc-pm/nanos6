/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "NUMAManager.hpp"
#include "dependencies/DataTrackingSupport.hpp"

#include <DataAccessRegistration.hpp>

NUMAManager::directory_t NUMAManager::_directory;
RWSpinLock NUMAManager::_lock;
NUMAManager::alloc_info_t NUMAManager::_allocations;
SpinLock NUMAManager::_allocationsLock;
NUMAManager::bitmask_t NUMAManager::_bitmaskNumaAll;
NUMAManager::bitmask_t NUMAManager::_bitmaskNumaAllActive;
NUMAManager::bitmask_t NUMAManager::_bitmaskNumaAnyActive;
std::atomic<bool> NUMAManager::_trackingEnabled;
ConfigVariable<bool> NUMAManager::_reportEnabled("numa.report");
ConfigVariable<std::string> NUMAManager::_trackingMode("numa.tracking");
ConfigVariable<bool> NUMAManager::_discoverPageSize("numa.discover");
size_t NUMAManager::_realPageSize;
int NUMAManager::_maxOSIndex;
std::vector<int> NUMAManager::_logicalToOsIndex;

#ifndef NDEBUG
void NUMAManager::checkAllocationCorrectness(
	void *res, size_t size,
	const bitmask_t *bitmask,
	size_t blockSize
) {
	size_t pageSize = _realPageSize;
	assert(pageSize > 0);

	unsigned long numPages = MathSupport::ceil(size, pageSize);
	assert(numPages > 0);

	void **pages = (void **) MemoryAllocator::alloc(numPages * sizeof(void *));
	assert(pages != nullptr);

	int *status = (int *) MemoryAllocator::alloc(numPages * sizeof(int));
	assert(status != nullptr);

	bitmask_t bitmaskCopy = *bitmask;
	uint8_t currentNodeIndex = BitManipulation::indexFirstEnabledBit(bitmaskCopy);
	BitManipulation::disableBit(&bitmaskCopy, currentNodeIndex);
	if (bitmaskCopy == 0) {
		bitmaskCopy = *bitmask;
	}

	size_t page = 0;
	size_t blockBytes = 0;
	for (size_t i = 0; i < size; i += pageSize) {
		if (blockBytes >= blockSize) {
			currentNodeIndex = BitManipulation::indexFirstEnabledBit(bitmaskCopy);
			BitManipulation::disableBit(&bitmaskCopy, currentNodeIndex);
			if (bitmaskCopy == 0) {
				bitmaskCopy = *bitmask;
			}
			blockBytes = 0;
		}

		char *tmp = (char *) res+i;
		// Fault the page, otherwise move_pages do not work. Writing 1 byte is enough.
		memset(tmp, 0, 1);
		pages[page] = tmp;

		blockBytes += pageSize;
		page++;
	}
	assert(numPages == page);

	{
		int pid = 0;
		int *nodes = nullptr;
		int flags = 0;
		long ret = move_pages(pid, numPages, pages, nodes, status, flags);
		assert(ret == 0);
	}

	// Check pages are properly distributed
	bitmaskCopy = *bitmask;
	currentNodeIndex = BitManipulation::indexFirstEnabledBit(bitmaskCopy);
	BitManipulation::disableBit(&bitmaskCopy, currentNodeIndex);
	if (bitmaskCopy == 0) {
		bitmaskCopy = *bitmask;
	}

	blockBytes = 0;
	for (size_t i = 0; i < numPages; i++) {
		if (blockBytes >= blockSize) {
			currentNodeIndex = BitManipulation::indexFirstEnabledBit(bitmaskCopy);
			BitManipulation::disableBit(&bitmaskCopy, currentNodeIndex);
			if (bitmaskCopy == 0) {
				bitmaskCopy = *bitmask;
			}
			blockBytes = 0;
		}

		assert(status[i] >= 0);
		FatalErrorHandler::warnIf(status[i] != _logicalToOsIndex[currentNodeIndex], "Page is not where it should.");

		blockBytes += pageSize;
	}

	MemoryAllocator::free(pages, numPages * sizeof(void *));
	MemoryAllocator::free(status, numPages * sizeof(int));
}
#endif

bool NUMAManager::isTrackingEnabled()
{
	return (_trackingEnabled.load(std::memory_order_relaxed) &&
			getValidTrackingNodes() > 1 &&
			DataAccessRegistration::supportsDataTracking());
}

uint64_t NUMAManager::getTrackingNodes()
{
	// This method is called from UnsyncScheduler::UnsyncScheduler()
	// before calling NUMAManager::initialize().
	std::string trackingMode = _trackingMode.getValue();
	if (trackingMode == "off" ||
			!DataAccessRegistration::supportsDataTracking() ||
			!DataTrackingSupport::isNUMASchedulingEnabled())
	{
		return 1;
	} else {
		return HardwareInfo::getMemoryPlaceCount(nanos6_host_device);
	}
}

void NUMAManager::discoverRealPageSize()
{
	// Discover pagesize
	// In systems with Transparent Huge Pages (THP), we cannot know the THP pagesize.
	// We try to discover it in this piece of code. Basically, we do an allocation aligned
	// to 4MB. Then we try to distribute it in blocks of pagesize. Using move_pages, we
	// check if the distribution is done as expected. Otherwise, it means THP is enabled.
	// If so, we check the size of the THP by looking at the first page that has been
	// allocated in a different NUMA node.

	// Set up parameters of the allocation
	size_t sizeAlloc = 16*1024*1024;
	size_t pageSize = HardwareInfo::getPageSize();
	void *tmp = nullptr;

	// Actually perform the aligned allocation
	int err = posix_memalign(&tmp, 4*1024*1024, sizeAlloc);
	FatalErrorHandler::failIf(err != 0);

	bitmask_t bitmaskCopy = _bitmaskNumaAnyActive;
	assert(BitManipulation::countEnabledBits(&bitmaskCopy) > 1);
	assert(_maxOSIndex > 0);
	struct bitmask *tmpBitmask = numa_bitmask_alloc(_maxOSIndex + 1);
	for (size_t i = 0; i < sizeAlloc; i += pageSize) {
		// Touch first page using current CPU, and then the rest using the other.
		// Thus, the first page that has a different NUMA id indicates us the
		// real page size.
		numa_bitmask_clearall(tmpBitmask);
		uint8_t currentNodeIndex = BitManipulation::indexFirstEnabledBit(bitmaskCopy);
		if (i == 0) {
			BitManipulation::disableBit(&bitmaskCopy, currentNodeIndex);
		}
		numa_bitmask_setbit(tmpBitmask, _logicalToOsIndex[currentNodeIndex]);
		numa_bind(tmpBitmask);

		void *pageStart = (void *) ((uintptr_t) tmp + i);
		memset(pageStart, 0, 1);
	}
	numa_bitmask_free(tmpBitmask);

	assert(pageSize > 0);
	unsigned long numPages = MathSupport::ceil(sizeAlloc, pageSize);
	assert(numPages > 0);

	void **pages = (void **) MemoryAllocator::alloc(numPages * sizeof(void *));
	assert(pages != nullptr);

	int *status = (int *) MemoryAllocator::alloc(numPages * sizeof(int));
	assert(status != nullptr);

	size_t page = 0;
	for (size_t i = 0; i < sizeAlloc; i += pageSize) {
		char *pageStart = (char *) tmp+i;
		pages[page++] = pageStart;
	}
	assert(numPages == page);

	{
		int pid = 0;
		int *nodes = nullptr;
		int flags = 0;
		__attribute__((unused)) long ret = move_pages(pid, numPages, pages, nodes, status, flags);
		assert(ret == 0);
	}

	// Check the first page with a different NUMA node
	for (size_t i = 1; i < numPages; i++) {
		assert(status[i] >= 0);

		if (status[i] != status[0]) {
			_realPageSize = i*pageSize;
			break;
		}
	}

	MemoryAllocator::free(pages, numPages * sizeof(void *));
	MemoryAllocator::free(status, numPages * sizeof(int));

	std::free(tmp);
}
