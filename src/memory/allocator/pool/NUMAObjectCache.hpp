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
	
	typedef std::deque<T *> pool_t;
	typedef struct {
		pool_t _pool;
		PaddedSpinLock<64> _lock;
	} NUMApool_t;
	
	std::deque<NUMApool_t> _NUMAPools;
	
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
	inline int fillCPUPool(size_t numaId, std::deque<T *> &pool, size_t requestedObjects)
	{
		std::lock_guard<PaddedSpinLock<64>> lock(_NUMAPools[numaId]._lock);
		pool_t &numaPool = _NUMAPools[numaId]._pool;
		
		size_t poolSize = numaPool.size();
		if (poolSize == 0) {
			return 0;
		}
		
		size_t nrObjects = std::min(requestedObjects, poolSize);
		std::move(numaPool.begin(), numaPool.begin() + nrObjects, std::front_inserter(pool));
		numaPool.erase(numaPool.begin(), numaPool.begin() + nrObjects);
		
		return nrObjects;
	}
	
	/** Method to return objects to a NUMA pool from a CPUNUMAObjectCache.
	 *
	 * This is typically called from a CPUNUMAObjectCache in order to return
	 * objects related with a different NUMA node than the one it belongs.
	 */
	void returnObjects(size_t numaId, std::deque<T *> &pool)
	{
		std::lock_guard<PaddedSpinLock<64>> lock(_NUMAPools[numaId]._lock);
		pool_t &numaPool = _NUMAPools[numaId]._pool;
		std::move(pool.begin(), pool.end(), std::back_inserter(numaPool));
		pool.erase(pool.begin(), pool.end());
	}
};

#endif /* __NUMA_OBJECT_CACHE_HPP__ */
