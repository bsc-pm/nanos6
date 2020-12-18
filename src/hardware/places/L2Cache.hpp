/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef L2_CACHE_HPP
#define L2_CACHE_HPP

#include "HardwareCache.hpp"
#include "L3Cache.hpp"

class L2Cache : public HardwareCache {
private:
	unsigned _associatedL3Id;

	static std::atomic<uint64_t> _accessedBytes;
	static std::atomic<uint64_t> _missedBytes;

public:
	L2Cache (unsigned id, unsigned L3Id, size_t cacheSize, size_t cacheLineSize)
		: HardwareCache(id, cacheSize, cacheLineSize, DataTrackingSupport::L2_LEVEL),
		_associatedL3Id(L3Id)
	{}

	virtual ~L2Cache() {}

	virtual inline bool isInclusive()
	{
		assert(0);
		return false;
	}

	virtual inline DataTrackingSupport::timestamp_t addDataAccess(size_t size, DataTrackingSupport::timestamp_t time, L3Cache * l3Cache, DataTrackingSupport::timestamp_t &L3Time)
	{
		assert(size > 0 && (l3Cache != nullptr || L3Time == DataTrackingSupport::NOT_PRESENT));
		if (l3Cache != nullptr) {
			assert(l3Cache->getId() == _associatedL3Id);
			if (l3Cache->isInclusive()) {
				//! Accesses inserted in L2 must also be inserted in L3.
				//! Evictions from L2 DO NOT go to L3.
				L3Time = l3Cache->addDataAccess(size, L3Time);
				return addDataAccess(size, time);
			}
			else {
				//! Only evictions from L2 go to L3, NOT the whole access.
				size_t l2CachedBytes = getCachedBytes(time, size);
				size_t increment = size - l2CachedBytes;
				incrementAccessedBytes(size);
				incrementMissedBytes(increment);
				if (increment == 0)
					increment = 1;
				size_t result = _bytesInCache.fetch_add(increment);
				size_t L3Size = 0;
				if (result+increment >= _cacheSize && increment > 1) {
					// We are evicting something that goes to L3.
					L3Size = result >= _cacheSize ? increment : (result + increment) - _cacheSize;
				}
				L3Time = l3Cache->addDataAccess(L3Size, L3Time);
				assert(result+increment <= _bytesInCache);
				return result+increment;
			}
		}

		return addDataAccess(size, time);
	}

	virtual inline DataTrackingSupport::timestamp_t addDataAccess(size_t size, DataTrackingSupport::timestamp_t time)
	{
		assert(size > 0 && time <= now());
		size_t cachedBytes = getCachedBytes(time, size);
		//! Increment only the size that is not in cache.
		size_t increment = size - cachedBytes;
		incrementAccessedBytes(size);
		incrementMissedBytes(increment);
		//! If the data access was already in cache and we are just updating it, increment only 1 instead of the size.
		if (increment == 0)
			increment = 1;
		size_t result = _bytesInCache.fetch_add(increment);
		assert(result+increment <= _bytesInCache);
		return result+increment;
	}

	virtual inline size_t getCachedBytes(DataTrackingSupport::timestamp_t time, size_t size) {
		if (time == DataTrackingSupport::NOT_PRESENT) {
			return 0;
		}

		DataTrackingSupport::timestamp_t window_end = now();
		DataTrackingSupport::timestamp_t window_start = window_end < _cacheSize ? 0 : window_end - _cacheSize;

		assert(time >= size);
		DataTrackingSupport::timestamp_t access_start = time - size;
		if (time <= window_start) {
			return 0;
		} else if (access_start < window_start) {
			assert(time > window_start);
			return time - window_start;
		} else {
			// This may happen because we have no synchronization in the update of
			// tracking info. Thus, it may happen that we see a given location and
			// a timeL2 from a different location.
			if (time > window_end) {
				return 0;
			}
			assert((time <= window_end) && (access_start >= window_start));
			return size;
		}
	}

	inline void setAssociatedL3Id(unsigned id)
	{
		_associatedL3Id = id;
	}

	inline unsigned getAssociatedL3Id() const
	{
		return _associatedL3Id;
	}

	static inline void incrementMissedBytes(size_t increment)
	{
		if (DataTrackingSupport::isTrackingReportEnabled())
			_missedBytes.fetch_add(increment, std::memory_order_relaxed);
	}

	static inline void incrementAccessedBytes(size_t increment)
	{
		if (DataTrackingSupport::isTrackingReportEnabled())
			_accessedBytes.fetch_add(increment, std::memory_order_relaxed);
	}

	static inline uint64_t getMissedBytes()
	{
		return _missedBytes;
	}

	static inline uint64_t getAccessedBytes()
	{
		return _accessedBytes;
	}
};

#endif // L2_CACHE_HPP
