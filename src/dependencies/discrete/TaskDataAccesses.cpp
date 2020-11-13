/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "TaskDataAccesses.hpp"
#include "scheduling/SchedulerInterface.hpp"

std::default_random_engine TaskDataAccesses::_randomEngine;

void TaskDataAccesses::computeNUMAAffinity(uint8_t &chosenNUMAid, Task *)
{
	if (_totalDataSize == 0 || !DataTrackingSupport::isNUMATrackingEnabled() || !DataTrackingSupport::isNUMASchedulingEnabled())
		return;

	int numNUMANodes = HardwareInfo::getValidMemoryPlaceCount(nanos6_host_device);
	size_t *bytesInNUMA = (size_t *) alloca(numNUMANodes * sizeof(size_t));
	std::memset(bytesInNUMA, 0, numNUMANodes * sizeof(size_t));

	forAll([&](void *, DataAccess *dataAccess) -> bool {
		//! If the dataAccess is weak it is not really read/written, so no action required.
		if (!dataAccess->isWeak()) {
			uint8_t NUMAid = dataAccess->getHomeNode();
			if (NUMAid != (uint8_t) -1) {
				assert(NUMAid < numNUMANodes);
				// Apply a bonus factor to RW accesses
				DataAccessType type = dataAccess->getType();
				bool rwAccess = (type != READ_ACCESS_TYPE) && (type != WRITE_ACCESS_TYPE);
				if (rwAccess) {
					bytesInNUMA[NUMAid] += dataAccess->getAccessRegion().getSize() * DataTrackingSupport::RW_BONUS_FACTOR;
				} else {
					bytesInNUMA[NUMAid] += dataAccess->getAccessRegion().getSize();
				}
			}
			return true;
		}
		return true;
	});

	size_t max = 0;
	size_t sanityCheck = 0;
	uint8_t chosen = (uint8_t) -1;
	std::uniform_int_distribution<uint8_t> _unif(0, 1);
	for (int i = 0; i < numNUMANodes; i++) {
		sanityCheck += bytesInNUMA[i];
		if (bytesInNUMA[i] > max) {
			max = bytesInNUMA[i];
			chosen = i;
		} else if (bytesInNUMA[i] == max && chosen != (uint8_t) i) {
			// Random returns either 0 or 1. If 0, we keep the old max, if 1, we update it.
			uint8_t update = _unif(_randomEngine);
			if (update) {
				chosen = i;
			}
		}
	}
	chosenNUMAid = chosen;
	// At least, it must be _totalDataSize, at most, 2*_totalDataSize because RW accesses count double.
	assert(sanityCheck >= _totalDataSize && sanityCheck <= 2*_totalDataSize);
}
