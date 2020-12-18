/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "TaskDataAccesses.hpp"
#include "scheduling/SchedulerInterface.hpp"

void TaskDataAccesses::trackDataLocation(CPU *cpu)
{
	if (!DataTrackingSupport::isTrackingEnabled() && _totalDataSize > 0) {
		return;
	}

	assert(cpu != nullptr);
	L2Cache *l2Cache = cpu->getL2Cache();
	L3Cache *l3Cache = cpu->getL3Cache();
	int L2Index = l2Cache->getId();
	int L3Index = l3Cache->getId();

	forAll([&](void *, DataAccess *dataAccess) -> bool {
		//! If the dataAccess is weak it is not really read/written, so no action required.
		if (!dataAccess->isWeak()) {
			size_t accessLength = dataAccess->getAccessRegion().getSize();
			DataTrackingSupport::DataTrackingInfo *trackingInfo = dataAccess->getTrackingInfo();
			DataTrackingSupport::timestamp_t L2Time = DataTrackingSupport::NOT_PRESENT;
			DataTrackingSupport::timestamp_t L3Time = DataTrackingSupport::NOT_PRESENT;
			DataTrackingSupport::location_t loc = trackingInfo->_location;

			L2Cache *dataL2 = (loc == DataTrackingSupport::UNKNOWN_LOCATION) ? nullptr : HardwareInfo::getL2Cache(loc);
			int dataL3Index = (dataL2 == nullptr) ? -1 : dataL2->getAssociatedL3Id();

			if (dataL3Index == L3Index) {
				L3Time = trackingInfo->_timeL3;
				if (loc == L2Index) {
					assert(l2Cache == dataL2);
					L2Time = trackingInfo->_timeL2;
				}

				//! If cache is inclusive, a dataAccess that is present in L3, must be present in L2.
				assert(!l3Cache->isInclusive() ||
					(((L2Time == DataTrackingSupport::NOT_PRESENT && L3Time == DataTrackingSupport::NOT_PRESENT) || loc != L2Index)) ||
					((L2Time != DataTrackingSupport::NOT_PRESENT && L3Time != DataTrackingSupport::NOT_PRESENT) || loc != L2Index));

				// This add the access in both L2 and L3
				L2Time = l2Cache->addDataAccess(accessLength, L2Time, l3Cache, L3Time);
				assert(L2Time != DataTrackingSupport::NOT_PRESENT && L3Time != DataTrackingSupport::NOT_PRESENT);
			} else {
				assert(loc != L2Index);
				// This add the access in both L2 and L3
				L2Time = l2Cache->addDataAccess(accessLength, L2Time, l3Cache, L3Time);
				assert(L2Time != DataTrackingSupport::NOT_PRESENT);
				//! If there is L3Cache, the dataAccess must be present.
				assert(l3Cache == nullptr || L3Time != DataTrackingSupport::NOT_PRESENT);
			}


			//! Set locality info in the dataAccess.
			assert(L2Time != DataTrackingSupport::NOT_PRESENT);
			assert(L2Time <= l2Cache->now());
			assert(l2Cache->getId() == (unsigned int) L2Index);
			assert((L3Time != DataTrackingSupport::NOT_PRESENT) == (l3Cache != nullptr));

			dataAccess->updateTrackingInfo(L2Index, L2Time, L3Time);
		}
		return true;
	});
}

void TaskDataAccesses::computeTaskAffinity(unsigned int &chosenL2id, unsigned int &chosenL3id)
{
	if (!(DataTrackingSupport::isTrackingEnabled() && SchedulerInterface::isLocalityEnabled())) {
		return;
	}

	// If data size is too small or too big, locality does not matter.
	if (_totalDataSize >= DataTrackingSupport::MIN_TRACKING_THRESHOLD && _totalDataSize <= DataTrackingSupport::MAX_TRACKING_THRESHOLD) {
		int numL2Cache = HardwareInfo::getNumL2Cache();

		// Reorder accesses depending on the location
		int numAccesses = getRealAccessNumber();
		int numSlots = std::min(numL2Cache, numAccesses);
		std::vector<DataAccess **> accessesPerCache;
		accessesPerCache.reserve(numSlots);
		std::vector<int> numAccessesPerCache(numSlots, 0);
		int cachesWithAccesses = 0;
		int remainingAccesses = numAccesses;

		forAll([&](void *, DataAccess *dataAccess) -> bool {
			//! If the dataAccess is weak it is not really read/written, so no action required.
			if (!dataAccess->isWeak()) {
				DataTrackingSupport::DataTrackingInfo *trackingInfo = dataAccess->getTrackingInfo();
				if (trackingInfo->_location != DataTrackingSupport::UNKNOWN_LOCATION) {
					if (numAccessesPerCache[cachesWithAccesses] == 0) {
						DataAccess **accesses = (DataAccess **) malloc(remainingAccesses * sizeof(DataAccess));
						accessesPerCache.push_back(accesses);
					}
					accessesPerCache[cachesWithAccesses][numAccessesPerCache[cachesWithAccesses]] = dataAccess;
					numAccessesPerCache[cachesWithAccesses]++;
					cachesWithAccesses++;
					remainingAccesses--;
					return true;
				}
			}
					remainingAccesses--;
			return true;
		});

		// TODO: Try to begin with the cache with highest number of accesses
		double maxScoreL3 = 0.0;
		for (int i = 0; i < cachesWithAccesses; i++) {
			uint32_t bytesL2 = 0;
			uint32_t bytesL3 = 0;
			double scoreL2 = 0.0;
			double scoreL3 = 0.0;
			uint32_t processedBytes = 0;
			uint32_t remainingBytes = _totalDataSize;
			double maxPossibleScore = (double) remainingBytes / (double) _totalDataSize;

			if (maxPossibleScore >= DataTrackingSupport::L2_THRESHOLD) {
				DataTrackingSupport::DataTrackingInfo *trackingInfo = accessesPerCache[i][0]->getTrackingInfo();
				assert(trackingInfo != nullptr);
				unsigned int L2id = trackingInfo->_location;
				L2Cache *l2cache = HardwareInfo::getL2Cache(L2id);
				unsigned int L3id = l2cache->getAssociatedL3Id();
				L3Cache *l3cache = HardwareInfo::getL3Cache(L3id);

				for (int j = 0; j < numAccessesPerCache[i]; j++) {
					DataAccess *dataAccess = accessesPerCache[i][j];
					trackingInfo = dataAccess->getTrackingInfo();
					DataTrackingSupport::timestamp_t timeL2 = trackingInfo->_timeL2;
					DataTrackingSupport::timestamp_t timeL3 = trackingInfo->_timeL3;
					uint32_t dataAccessLength = dataAccess->getAccessRegion().getSize();
					maxPossibleScore = (double) remainingBytes / (double) _totalDataSize;

					// Check if task could reach the thresholds. Otherwise, discard it.
					if (scoreL2 + maxPossibleScore < DataTrackingSupport::L2_THRESHOLD) {
						break;
					}

					remainingBytes -= dataAccessLength;
					processedBytes += dataAccessLength;

					size_t cachedBytesL2 = l2cache->getCachedBytes(timeL2, dataAccessLength);
					assert(cachedBytesL2 <= dataAccessLength);

					// Update L2 score only if it can be good enough
					if (cachedBytesL2 > 0) {
						bytesL2 += cachedBytesL2;
						assert(bytesL2 <= processedBytes);
						scoreL2 += (double) bytesL3 / _totalDataSize;

						//! Cutoff good enough
						if (scoreL2 >= DataTrackingSupport::L2_THRESHOLD) {
							chosenL2id = L2id;
							chosenL3id = L3id;
							assert(scoreL2 <= 1.0 && scoreL2 >= 0.0);
							return;
						}
					}

					// Check if task could reach the thresholds. Otherwise, discard it.
					if (scoreL2 + maxPossibleScore/L3Cache::getPenalty() < DataTrackingSupport::L2_THRESHOLD &&
						scoreL3 + maxPossibleScore < DataTrackingSupport::L3_THRESHOLD)
					{
						break;
					}

					if (cachedBytesL2 < dataAccessLength) {
						size_t cachedBytesL3 = 0;
						cachedBytesL3 = l3cache->getCachedBytes(timeL3, dataAccessLength);
						if (!l3cache->isInclusive()) {
							// If L3 is non-inclusive, a block can be evicted from L3 without evicting itself from L2.
							// Our tracking is not perfect, it is just an approximation. Therefore, it may happen that
							// a block has been actually evicted from L2 but we just partially evicted it. Therefore,
							// it may happen that the aggregated bytes cached in L2 and L3 is bigger than the actual
							// length of the dataAccess. For instance, if the whole dataAccess is cached in L3 and we think
							// there are still 2 bytes of it in L2. We want to consider only the bytes missing in L2
							// based on our approximation. This cannot happen at all in inclusive L3 because an eviction
							// from L3 causes an eviction from L2.
							size_t maxCachedBytesL3 = dataAccessLength - cachedBytesL2;
							if (cachedBytesL3 > maxCachedBytesL3) {
								cachedBytesL3 = maxCachedBytesL3;
							}
						}
						assert(cachedBytesL3 + cachedBytesL2 <= dataAccessLength);

						if (cachedBytesL3 > 0) {
							bytesL2 += cachedBytesL3 / L3Cache::getPenalty();
							assert(bytesL2 <= processedBytes);
							scoreL2 += (double) bytesL3 / _totalDataSize;

							//! Cutoff good enough
							if (scoreL2 >= DataTrackingSupport::L2_THRESHOLD) {
								chosenL2id = L2id;
								chosenL3id = L3id;
								assert(scoreL2 <= 1.0 && scoreL2 >= 0.0);
								return;
							}

							bytesL3 += cachedBytesL3;
							assert(bytesL3 <= processedBytes);
							scoreL3 += (double) bytesL3 / _totalDataSize;

							//! Cannot cutoff because maybe next dataAccesses make the task schedulable to L2 local queues.
							if (scoreL3 >= DataTrackingSupport::L3_THRESHOLD && scoreL3 >= maxScoreL3) {
								chosenL3id = L3id;
								maxScoreL3 = scoreL3;
								assert(scoreL3 <= 1.0 && scoreL3 >= 0.0);
							}
						}
					}
				}
			}
			free(accessesPerCache[i]);
		}
	}
}

bool TaskDataAccesses::checkExpiration(unsigned int &chosenL2id, unsigned int &chosenL3id)
{
	if (!(DataTrackingSupport::isTrackingEnabled() &&
				SchedulerInterface::isLocalityEnabled() &&
				DataTrackingSupport::isCheckExpirationEnabled()))
	{
		return false;
	}

	// In this method we want to check if the task still has enough data to be run where it was enqueued
	// chosenL2id and chosenL3id contain the id of the queue where the task was initially enqueued
	// Check expiration only from L2.
	if (chosenL2id == (unsigned int) -1) {
		return false;
	}

	if (_totalDataSize >= DataTrackingSupport::MIN_TRACKING_THRESHOLD && _totalDataSize <= DataTrackingSupport::MAX_TRACKING_THRESHOLD) {
		size_t remainingBytes = _totalDataSize;
		double scoreL2 = 0.0;
		double scoreL3 = 0.0;
		size_t processedBytes = 0;
		bool expired = false;

		forAll([&](void *, DataAccess *dataAccess) {
			//! If the dataAccess is weak it is not really read/written, so no action required.
			if (!dataAccess->isWeak()) {
				size_t dataAccessLength = dataAccess->getAccessRegion().getSize();
				double maxPossibleScore = (double) remainingBytes / (double) _totalDataSize;

				// Check if task could reach the thresholds. Otherwise, discard it.
				if (scoreL3 / (double) _totalDataSize + maxPossibleScore < DataTrackingSupport::L3_THRESHOLD && scoreL2 / (double) _totalDataSize + maxPossibleScore < DataTrackingSupport::L2_THRESHOLD) {
					chosenL2id = -1;
					chosenL3id = -1;
					expired = true;
					// Break forAll
					return false;
				}

				remainingBytes -= dataAccessLength;
				processedBytes += dataAccessLength;
				assert(processedBytes + remainingBytes == _totalDataSize);

				DataTrackingSupport::DataTrackingInfo *trackingInfo = dataAccess->getTrackingInfo();
				DataTrackingSupport::location_t loc = (trackingInfo != nullptr) ? trackingInfo->_location : DataTrackingSupport::UNKNOWN_LOCATION;

				if (loc != DataTrackingSupport::UNKNOWN_LOCATION) {
					DataTrackingSupport::timestamp_t timeL2 = trackingInfo->_timeL2;
					DataTrackingSupport::timestamp_t timeL3 = trackingInfo->_timeL3;
					unsigned int L2id = loc;
					L2Cache *l2cache = HardwareInfo::getL2Cache(loc);
					unsigned int L3id = l2cache->getAssociatedL3Id();
					L3Cache *l3cache = HardwareInfo::getL3Cache(L3id);
					size_t cachedBytesL2 = 0;

					// Update only if the access is in the same place where it was initially enqueued
					if (L2id == chosenL2id) {
						cachedBytesL2 = l2cache->getCachedBytes(timeL2, dataAccessLength);
						assert(cachedBytesL2 <= dataAccessLength);

						// Update L2 score only if it can be good enough
						if (cachedBytesL2 > 0 && (scoreL2 + maxPossibleScore >= DataTrackingSupport::L2_THRESHOLD)) {
							scoreL2 += cachedBytesL2;
							assert(scoreL2 <= processedBytes);
							double score = scoreL2 / _totalDataSize;

							//! Cutoff good enough
							if (score >= DataTrackingSupport::L2_THRESHOLD) {
								expired = false;
								assert(score <= 1.0 && score >= 0.0);
								// Break forAll
								return false;
							}
						}
					}

					if (L3id == chosenL3id) {
						size_t cachedBytesL3 = 0;
						cachedBytesL3 = l3cache->getCachedBytes(timeL3, dataAccessLength);
						if (!l3cache->isInclusive()) {
							// If L3 is non-inclusive, a block can be evicted from L3 without evicting itself from L2.
							// Our tracking is not perfect, it is just an approximation. Therefore, it may happen that
							// a block has been actually evicted from L2 but we just partially evicted it. Therefore,
							// it may happen that the aggregated bytes cached in L2 and L3 is bigger than the actual
							// length of the dataAccess. For instance, if the whole dataAccess is cached in L3 and we think
							// there are still 2 bytes of it in L2. We want to consider only the bytes missing in L2
							// based on our approximation. This cannot happen at all in inclusive L3 because an eviction
							// from L3 causes an eviction from L2.
							size_t maxCachedBytesL3 = dataAccessLength - cachedBytesL2;
							if (cachedBytesL3 > maxCachedBytesL3) {
								cachedBytesL3 = maxCachedBytesL3;
							}
						}
						assert(cachedBytesL3 + cachedBytesL2 <= dataAccessLength);

						if (cachedBytesL3 > 0) {
							scoreL3 += cachedBytesL3;

							// Check if considering L3 bytes, we can still reach DataTrackingSupport::L2_THRESHOLD in some L2
							if (scoreL2 / (double) _totalDataSize + maxPossibleScore >= DataTrackingSupport::L2_THRESHOLD) {
								if (l2cache->getAssociatedL3Id() == L3id) {

									if (!l3cache->isInclusive()) {
										scoreL2 += cachedBytesL3 / L3Cache::getPenalty();
									} else {
										__attribute__((unused)) size_t maxCachedBytesL3 = dataAccessLength - cachedBytesL2;
										//! In inclusive caches, cachedBytesL3 and cachedBytesL2 may have the same data cached.
										//! Therefore, what makes the difference is the extra bytes cached by L3 compared with L2.
										assert(cachedBytesL3 >= cachedBytesL2);
										size_t additionalCachedBytesL3 = cachedBytesL3 - cachedBytesL2;
										assert(additionalCachedBytesL3 <= maxCachedBytesL3);
										cachedBytesL3 = additionalCachedBytesL3;
									}

									assert(scoreL2 <= processedBytes);
									double score = scoreL2 / _totalDataSize;

									//! Cutoff good enough
									if (score >= DataTrackingSupport::L2_THRESHOLD) {
										expired = false;
										assert(score <= 1.0 && score >= 0.0);
										// Break forAll
										return false;
									}
								}
							}
						}

						double score = scoreL3 / _totalDataSize;
						//! Cannot cutoff because maybe next dataAccesses make the task schedulable to L2 local queues.
						if (score >= DataTrackingSupport::L3_THRESHOLD) {
							assert(score <= 1.0 && score >= 0.0);
							if (scoreL2 + maxPossibleScore < DataTrackingSupport::L2_THRESHOLD) {
								chosenL2id = -1;
								expired = true;
								// Break forAll
								return false;
							}
						}
					}
				}
			}

			return true;
		});
		assert(chosenL2id == (unsigned int) -1 || expired == false);
		return expired;
	}

	return false;
}

void TaskDataAccesses::computeNUMAAffinity(uint8_t &chosenNUMAid)
{
	int numNUMAnodes = HardwareInfo::getValidMemoryPlaceCount(nanos6_host_device);
	int numAccesses = getRealAccessNumber();
	int numSlots = std::min(numNUMAnodes, numAccesses);
	size_t *bytesInNUMA = (size_t *) alloca(numSlots * sizeof(size_t));
    std::memset(bytesInNUMA, 0, numSlots * sizeof(size_t));

	forAll([&](void *, DataAccess *dataAccess) -> bool {
		//! If the dataAccess is weak it is not really read/written, so no action required.
		if (!dataAccess->isWeak()) {
			uint8_t NUMAid = dataAccess->getHomeNode();
			if (NUMAid != (uint8_t) -1) {
				assert(NUMAid < numNUMAnodes);
				bytesInNUMA[NUMAid] += dataAccess->getAccessRegion().getSize();
			}
			return true;
		}
		return true;
	});

	size_t max = 0;
	uint8_t chosen = (uint8_t) -1;
	for (int i = 0; i < numSlots; i++) {
		if (bytesInNUMA[i] > max) {
			max = bytesInNUMA[i];
			chosen = i;
		}
	}
	chosenNUMAid = chosen;
}
