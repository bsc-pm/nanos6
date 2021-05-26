/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef L3_CACHE_HPP
#define L3_CACHE_HPP

#include "HardwareCache.hpp"

#ifndef L3_DEFAULT_CACHE_SIZE
#define L3_DEFAULT_CACHE_SIZE 32*1024*1024
#endif // L3_DEFAULT_CACHE_SIZE

class L3Cache : public HardwareCache {
private:
	bool _inclusive;

public:
	L3Cache (int id, size_t cacheSize, size_t cacheLineSize, bool inclusive) :
		HardwareCache(id, cacheSize, cacheLineSize, HardwareCache::L3_LEVEL),
		_inclusive(inclusive)
	{
	}

	~L3Cache()
	{
	}

	inline bool isInclusive() const
	{
		return _inclusive;
	}
};

#endif // L3_CACHE_HPP
