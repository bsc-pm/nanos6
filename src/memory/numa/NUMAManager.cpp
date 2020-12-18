/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "NUMAManager.hpp"

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

#ifndef NDEBUG
void NUMAManager::checkAllocationCorrectness(
	void *res, size_t size,
	const bitmask_t *bitmask,
	size_t blockSize
) {
	size_t pageSize = HardwareInfo::getPageSize();
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
		tmp[0] = 0;
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
		FatalErrorHandler::warnIf(status[i] != currentNodeIndex, "Page is not where it should.");

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
	if (trackingMode == "off" || !DataAccessRegistration::supportsDataTracking()) {
		return 1;
	} else {
		return HardwareInfo::getMemoryPlaceCount(nanos6_host_device);
	}
}
