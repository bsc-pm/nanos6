/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_PLACE_HPP
#define CPU_PLACE_HPP

#include <atomic>

#include "ComputePlace.hpp"
#include "L2Cache.hpp"
#include "L3Cache.hpp"

class CPUPlace: public ComputePlace {
private:
	L2Cache *_l2cache;
	L3Cache *_l3cache;

public:
	CPUPlace(int index, bool owned = true) :
		ComputePlace(index, nanos6_device_t::nanos6_host_device, owned)
	{
    }

	CPUPlace (int index, L2Cache *l2cache, L3Cache *l3cache, bool owned = true)
		: ComputePlace(index, nanos6_device_t::nanos6_host_device, owned), _l2cache(l2cache), _l3cache(l3cache)
	{
		assert(l2cache != nullptr);
		l2cache->addCPU((CPU *) this);

		//! L3Cache is not mandatory. For instance, KNL in flat mode has no L3.
		if(l3cache != nullptr)
			l3cache->addCPU((CPU *) this);
	}

	~CPUPlace()
	{}

	inline bool hasL3Cache()
	{
		return _l3cache != nullptr;
	}

	inline void resetL2Cache()
	{
		_l2cache->reset();
	}

	inline void resetL3Cache()
	{
		_l3cache->reset();
	}

	inline bool isL2Full()
	{
		return _l2cache->isFull();
	}

	inline bool isL3Full()
	{
		return _l3cache->isFull();
	}

	inline DataTrackingSupport::timestamp_t addL3DataAccess(size_t size, DataTrackingSupport::timestamp_t time)
	{
		if(_l3cache != nullptr)
			return _l3cache->addDataAccess(size, time);
		assert(0);
		return DataTrackingSupport::NOT_PRESENT;
	}

	inline size_t getL2CachedBytes(DataTrackingSupport::timestamp_t time, size_t size)
	{
		return _l2cache->getCachedBytes(time, size);
	}

	inline size_t getL3CachedBytes(DataTrackingSupport::timestamp_t time, size_t size)
	{
		if(_l3cache != nullptr)
			return _l3cache->getCachedBytes(time, size);
		return 0;
	}

	inline void setL2Cache(L2Cache *l2cache)
	{
		_l2cache = l2cache;
	}

	inline L2Cache *getL2Cache() const
	{
		return _l2cache;
	}

	inline unsigned getL2CacheId() const
	{
		return _l2cache->getId();
	}

	inline size_t getL2CacheSize() const
	{
		return _l2cache->getCacheSize();
	}

	inline size_t getL2CacheLineSize() const
	{
		return _l2cache->getCacheLineSize();
	}

	inline void setL3Cache(L3Cache *l3cache)
	{
		_l3cache = l3cache;
	}

	inline L3Cache *getL3Cache() const
	{
		return _l3cache;
	}

	inline unsigned getL3CacheId() const
	{
		if(_l3cache != nullptr)
			return _l3cache->getId();
		assert(0 && "There must be a L3 cache to get the id.");
		return 0;
	}

	inline size_t getL3CacheSize() const
	{
		if(_l3cache != nullptr)
			return _l3cache->getCacheSize();
		assert(0 && "There must be a L3 cache to get the size.");
		return 0;
	}

	inline size_t getL3CacheLineSize() const
	{
		if(_l3cache != nullptr)
			return _l3cache->getCacheLineSize();
		assert(0 && "There must be a L3 cache to get the line size.");
		return 0;
	}

	inline bool isL3CacheInclusive()  const
	{
		if(_l3cache != nullptr)
			return _l3cache->isInclusive();
		assert(0 && "There must be a L3 cache to get the inclusiveness.");
		return false;
	}
};

#endif // CPU_PLACE_HPP
