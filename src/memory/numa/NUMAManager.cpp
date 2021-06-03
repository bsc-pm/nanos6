/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2021 Barcelona Supercomputing Center (BSC)
*/

#include <fstream>

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
ConfigVariable<bool> NUMAManager::_discoverPageSize("numa.discover_pagesize");
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

size_t NUMAManager::discoverRealPageSize()
{
	// In systems with Transparent Huge Pages (THP), we cannot know the real THP
	// pagesize. To discover it, we try to obtain it from kernel files. Ultimately,
	// we try to discover it using /proc/meminfo. If it is unobtainable using
	// the previous methods, we fallback to a default value reported by sysconf

	size_t pagesize = 0;
	std::ifstream enabledFile("/sys/kernel/mm/transparent_hugepage/enabled");
	if (enabledFile.is_open()) {
		std::string line;
		std::getline(enabledFile, line);
		int first = line.find('[');
		int last  = line.find(']');
		std::string policyString = line.substr(first+1, last-first-1);

		// If the policy is always (THP always enabled), we try to obtain the
		// page size using files, otherwise we fallback to the default value
		if (policyString == "always") {
			std::ifstream pagesizeFile("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size");
			if (pagesizeFile.is_open()) {
				pagesizeFile >> pagesize;
			} else {
				std::ifstream meminfoFile("/proc/meminfo");
				if (meminfoFile.is_open()) {
					while (std::getline(meminfoFile, line)) {
						if (line.find("Hugepagesize") != std::string::npos) {
							first = line.find(':');
							last  = line.find("kB");
							std::string psString = line.substr(first+1, last-first-1);
							psString.erase(std::remove(psString.begin(), psString.end(), ' '), psString.end());
							pagesize = strtoull(psString.c_str(), nullptr, 0) * 1024; // Assuming kB
						}
					}
				}
			}
		}
	}

	if (pagesize == 0) {
		pagesize = HardwareInfo::getPageSize();
		FatalErrorHandler::warn("Could not determine whether THP is enabled. Using default pagesize.");
	}
	assert(pagesize > 0);

	return pagesize;
}

