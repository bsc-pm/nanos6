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

public:
	L3Cache (unsigned id, size_t cacheSize, size_t cacheLineSize, bool inclusive)
		: HardwareCache(id, cacheSize, cacheLineSize, DataTrackingSupport::L3_LEVEL), _inclusive(inclusive)
	{}

	virtual ~L3Cache() {}

	virtual inline bool isInclusive()
	{
		return _inclusive;
	}
};

#endif // L3_CACHE_HPP
