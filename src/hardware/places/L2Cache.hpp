/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef L2_CACHE_HPP
#define L2_CACHE_HPP

#include "HardwareCache.hpp"

class L3Cache;

class L2Cache : public HardwareCache {
private:
	L3Cache *_associatedL3Cache;

public:
	L2Cache (int id, L3Cache *l3Cache, size_t cacheSize, size_t cacheLineSize) :
		HardwareCache(id, cacheSize, cacheLineSize, HardwareCache::L2_LEVEL),
		_associatedL3Cache(l3Cache)
	{
	}

	~L2Cache()
	{
	}

	inline void setAssociatedL3Cache(L3Cache *l3Cache)
	{
		_associatedL3Cache = l3Cache;
	}

	inline L3Cache *getAssociatedL3Cache() const
	{
		return _associatedL3Cache;
	}
};

#endif // L2_CACHE_HPP
