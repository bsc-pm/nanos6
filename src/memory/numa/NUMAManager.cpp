/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2021 Barcelona Supercomputing Center (BSC)
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
	// In systems with Transparent Huge Pages (THP), we cannot know the real THP
	// pagesize. To discover it, we allocate 32MB (16MB aligned to 16MB) and
	// distribute the allocation in blocks of "pageSize". Using 'move_pages' we
	// check if the distribution is done as expected. If it's not, it means THP
	// is enabled, and we know the real size by checking the first page that is
	// allocated in a different NUMA node

	// Set up parameters of the allocation (32MB)
	size_t sizeAlloc = 32*1024*1024;
	int prot = PROT_READ | PROT_WRITE;
	int flags = MAP_ANONYMOUS | MAP_PRIVATE | MAP_NORESERVE | MAP_NONBLOCK;
	int fd = -1;
	int offset = 0;

	// We use mmap as it returns virtual memory never allocated beforehand.
	// 'malloc' or 'posix_memalign' could return recycled memory
	void *tmp = mmap(nullptr, sizeAlloc, prot, flags, fd, offset);
	FatalErrorHandler::failIf(tmp == MAP_FAILED, "Couldn't allocate memory.");

	// We do this to obtain a pointer from 'tmp' aligned to 16MB
	size_t sizeAligned = 16*1024*1024;
	void *alignedTmp = (void *) (((uintptr_t) tmp + sizeAligned - 1) & ((uintptr_t) ~(sizeAligned - 1)));

	// Page size returned by the system, we will check if THP is enabled
	size_t pageSize = HardwareInfo::getPageSize();
	assert(pageSize > 0);

	bitmask_t bitmaskCopy = _bitmaskNumaAnyActive;
	assert(BitManipulation::countEnabledBits(&bitmaskCopy) > 1);

	// Before changing the memory policy and binding, retreive their initial
	// values to reset them later
	int previousMemoryMode;
	unsigned long previousMemoryNodemask;
	size_t maxNumNUMANodes = numa_num_possible_nodes();
	bitmask *previousMembindBitmask = MemoryAllocator::alloc(sizeof(bitmask));
	previousMembindBitmask = numa_get_membind();
	long retValue = get_mempolicy(&previousMemoryMode, &previousMemoryNodemask, maxNumNUMANodes, nullptr, 0);
	FatalErrorHandler::failIf(retValue != 0, "Couldn't obtain the current memory policy");

	// In this loop we visit the allocation of 16MB in chunks of 'pageSize'.
	// We bind the current CPU to the first NUMA node and modify the first
	// page-chunk. Then, we modify the rest of pages while binding the CPU
	// to the second NUMA node. If THP is disabled, the second page will
	// reside in the second NUMA node. Otherwise, more than just the first
	// page will reside in the first NUMA node
	bitmask *tmpBitmask = numa_bitmask_alloc(maxNumNUMANodes);
	for (size_t i = 0; i < sizeAligned; i += pageSize) {
		// If it's the first NUMA node, delete it from the mask as we
		// only want one page in this NUMA node
		numa_bitmask_clearall(tmpBitmask);
		uint8_t currentNodeIndex = BitManipulation::indexFirstEnabledBit(bitmaskCopy);
		if (i == 0) {
			BitManipulation::disableBit(&bitmaskCopy, currentNodeIndex);
		}

		// Prepare the current mask with the current NUMA node
		numa_bitmask_setbit(tmpBitmask, _logicalToOsIndex[currentNodeIndex]);

		// Make sure we're using a strict policy to ensure the mask
		retValue = set_mempolicy(MPOL_BIND, tmpBitmask->maskp, maxNumNUMANodes);
		FatalErrorHandler::failIf(retValue != 0, "Couldn't set a strict memory policy");

		// Bind the current CPU to the current mask
		numa_bind(tmpBitmask);

		// Modify the current page-chunk
		void *pageStart = (void *) ((uintptr_t) alignedTmp + i);
		memset(pageStart, 0, 1);
	}

	// Reset the memory policy and binding to their previous values
	numa_set_membind(previousMembindBitmask);
	retValue = set_mempolicy(&previousMemoryMode, &previousMemoryNodemask, maxNumNUMANodes);
	FatalErrorHandler::failIf(retValue != 0, "Couldn't reset the memory policy");

	// Prepare pointers to each page-chunk for the 'move_pages' call
	size_t numPages = MathSupport::ceil(sizeAligned, pageSize);
	void **pages = (void **) MemoryAllocator::alloc(numPages * sizeof(void *));
	int *status  = (int *)   MemoryAllocator::alloc(numPages * sizeof(int));
	assert(pages != nullptr);
	assert(status != nullptr);

	size_t page = 0;
	for (size_t i = 0; i < sizeAligned; i += pageSize) {
		char *pageStart = (char *) alignedTmp + i;
		pages[page++] = pageStart;
	}

	// Move the pages
	{
		__attribute__((unused)) long ret = move_pages(0, numPages, pages, nullptr, status, 0);
		assert(ret == 0);
	}

	// Check which is the first page residing in a different NUMA node
	for (size_t i = 1; i < numPages; i++) {
		assert(status[i] >= 0);
		if (status[i] != status[0]) {
			_realPageSize = i * pageSize;
			break;
		}
	}

	// Free all the unnecessary structures
	retValue = munmap(tmp, sizeAlloc);
	FatalErrorHandler::failIf(retValue != 0, "Couldn't unmap a memory region.");
	numa_bitmask_free(tmpBitmask);
	MemoryAllocator::free(pages, numPages * sizeof(void *));
	MemoryAllocator::free(status, numPages * sizeof(int));

	// In the event that we have not found a page with a different NUMA node,
	// it may happen that the system has pages larger than 32MB. In this
	// scenario, warn the user that we are using a default size
	if (_realPageSize == 0) {
		FatalErrorHandler::warn("Could not determine whether the sizing of Transparent Huge Pages.");
		FatalErrorHandler::warn("Using default page size, which may impace NUMA awareness performance.");
		_realPageSize = pageSize;
	}
	assert(_realPageSize > 0);
}

