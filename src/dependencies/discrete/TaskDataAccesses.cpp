/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "TaskDataAccesses.hpp"
#include "memory/numa/NUMAManager.hpp"
#include "scheduling/SchedulerInterface.hpp"

uint64_t TaskDataAccesses::computeNUMAAffinity(ComputePlace *computePlace)
{
	if (_totalDataSize == 0 ||
		!NUMAManager::isTrackingEnabled() ||
		!DataTrackingSupport::isNUMASchedulingEnabled())
	{
		return (uint64_t) -1;
	}

	size_t numNUMANodes = HardwareInfo::getMemoryPlaceCount(nanos6_host_device);
	size_t *bytesInNUMA = computePlace->getDependencyData()._bytesInNUMA;
	assert(bytesInNUMA != nullptr);

	// Init bytesInNUMA to zero
	std::memset(bytesInNUMA, 0, numNUMANodes * sizeof(size_t));

	std::minstd_rand0 &randomEngine = computePlace->getRandomEngine();
	size_t max = 0;
	uint64_t chosen = (uint64_t) -1;

	forAll([&](void *, const DataAccess *dataAccess) -> bool {
		//! If the dataAccess is weak it is not really read/written, so no action required.
		if (!dataAccess->isWeak()) {
			uint8_t numaId = dataAccess->getHomeNode();
			if (numaId != (uint8_t) -1) {
				assert(numaId < numNUMANodes);
				// Apply a bonus factor to RW accesses
				DataAccessType type = dataAccess->getType();
				bool rwAccess = (type != READ_ACCESS_TYPE) && (type != WRITE_ACCESS_TYPE);
				if (rwAccess) {
					bytesInNUMA[numaId] += dataAccess->getLength() * DataTrackingSupport::getRWBonusFactor();
				} else {
					bytesInNUMA[numaId] += dataAccess->getLength();
				}

				if (bytesInNUMA[numaId] > max) {
					max = bytesInNUMA[numaId];
					chosen = numaId;
				} else if (bytesInNUMA[numaId] == max && chosen != numaId) {
					// Random returns either 0 or 1. If 0, we keep the old max, if 1, we update it.
					std::uniform_int_distribution<unsigned int> unif(0, 1);
					unsigned int update = unif(randomEngine);
					if (update) {
						chosen = numaId;
					}
				}
			}
		}
		return true;
	});

	return chosen;
}
