/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef __NUMA_OBJECT_CACHE_HPP__
#define __NUMA_OBJECT_CACHE_HPP__

#include "lowlevel/PaddedSpinLock.hpp"

#include <deque>
#include <mutex>

template <typename T>
class NUMAObjectCache {
	size_t _NUMANodeCount;
	
	typedef std::deque<T *> Pool_t;
	typedef struct {
		Pool_t _pool;
		PaddedSpinLock<64> _lock;
	} NUMAPool_t;
	
	std::deque<NUMAPool_t> _NUMAPools;
	
public:
	NUMAObjectCache(size_t NUMANodeCount)
		: _NUMANodeCount(NUMANodeCount)
	{
		_NUMAPools.resize(_NUMANodeCount + 1);
	}
	
	~NUMAObjectCache()
	{
	}
	
	/** This is called from a CPUNUMAObjectCache to fill up its pool of
	 *  objects.
	 *
	 * The NUMAObjectCache will try to get objects from the NUMA-specific
	 * pool. The method returns how many objects it managed to ultimately
	 * allocate
	 */
	inline int fillCPUPool(size_t NUMAId, std::deque<T *> &pool, size_t requestedObjects)
	{
		std::lock_guard<PaddedSpinLock<64>> lock(_NUMAPools[NUMAId]._lock);
		Pool_t &NUMAPool = _NUMAPools[NUMAId]._pool;
		
		size_t poolSize = NUMAPool.size();
		if (poolSize == 0) {
			return 0;
		}
		
		size_t nrObjects = std::max(requestedObjects, poolSize);
		std::move(NUMAPool.begin(), NUMAPool.begin() + nrObjects, pool.begin());
		
		return nrObjects;
	}
	
	/** Method to return objects to a NUMA pool from a CPUNUMAObjectCache.
	 *
	 * This is typically called from a CPUNUMAObjectCache in order to return
	 * objects related with a different NUMA node than the one it belongs.
	 */
	void returnObjects(size_t NUMAId, std::deque<T *> &pool)
	{
		std::lock_guard<PaddedSpinLock<64>> lock(_NUMAPools[NUMAId]._lock);
		Pool_t &NUMAPool = _NUMAPools[NUMAId]._pool;
		std::move(pool.begin(), pool.end(), NUMAPool.end());
	}
};

#endif /* __NUMA_OBJECT_CACHE_HPP__ */
