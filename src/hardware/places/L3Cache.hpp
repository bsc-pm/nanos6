/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef L3_CACHE_HPP
#define L3_CACHE_HPP

#include "HardwareCache.hpp"

class L3Cache : public HardwareCache {
private:
	bool _inclusive;
	static double _penalty;

	static std::atomic<uint64_t> _accessedBytes;
	static std::atomic<uint64_t> _missedBytes;

public:
	L3Cache (unsigned id, size_t cacheSize, size_t cacheLineSize, bool inclusive)
		: HardwareCache(id, cacheSize, cacheLineSize, DataTrackingSupport::L3_LEVEL), _inclusive(inclusive)
	{}

	virtual ~L3Cache() {}

	virtual inline bool isInclusive()
	{
		return _inclusive;
	}

	virtual inline DataTrackingSupport::timestamp_t addDataAccess(size_t size, DataTrackingSupport::timestamp_t time)
	{
		size_t cachedBytes = getCachedBytes(time, size);
		//! Increment only the size that is not in cache.
		size_t increment = size - cachedBytes;
		incrementAccessedBytes(size);
		incrementMissedBytes(increment);
		//! If the data access was already in cache and we are just updating it, increment only 1 instead of the size.
		if(increment == 0)
			increment = 1;
		size_t result = _bytesInCache.fetch_add(increment);

		assert(result+increment <= _bytesInCache);
		return result+increment;
	}

	virtual inline size_t getCachedBytes(DataTrackingSupport::timestamp_t time, size_t size) {
		if(time == DataTrackingSupport::NOT_PRESENT) {
			return 0;
		}

		DataTrackingSupport::timestamp_t window_end = now();
		DataTrackingSupport::timestamp_t window_start = window_end < _cacheSize ? 0 : window_end - _cacheSize;

		DataTrackingSupport::timestamp_t access_start = time - size;
		//DataTrackingSupport::timestamp_t access_start = size > time ? 0 : time - size;
		if(time <= window_start) {
			return 0;
		} else if(access_start < window_start) {
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

	virtual inline DataTrackingSupport::timestamp_t addDataAccess(__attribute__((unused)) size_t size, __attribute__((unused)) DataTrackingSupport::timestamp_t time,
											 __attribute__((unused)) L3Cache * l3Cache, __attribute__((unused)) DataTrackingSupport::timestamp_t &L3Time)
	{
		//! Pure virtual function. Must be defined but must be never called in this specialization.
		assert(0);
		return DataTrackingSupport::NOT_PRESENT;
	}

	static inline void setPenalty(double penalty)
	{
		_penalty = penalty;
	}

	static inline double getPenalty()
	{
		return _penalty;
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

#endif // L3_CACHE_HPP
