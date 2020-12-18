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

public:
	L2Cache (unsigned id, unsigned L3Id, size_t cacheSize, size_t cacheLineSize)
		: HardwareCache(id, cacheSize, cacheLineSize, DataTrackingSupport::L2_LEVEL),
		_associatedL3Id(L3Id)
	{}

	virtual ~L2Cache() {}

	inline void setAssociatedL3Id(unsigned id)
	{
		_associatedL3Id = id;
	}

	inline unsigned getAssociatedL3Id() const
	{
		return _associatedL3Id;
	}
};

#endif // L2_CACHE_HPP
