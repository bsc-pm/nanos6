/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef __CPU_OBJECT_CACHE_HPP__
#define __CPU_OBJECT_CACHE_HPP__

#include "lowlevel/SpinLock.hpp"
#include <deque>

#include <NUMAObjectCache.hpp>
#include <VirtualMemoryManagement.hpp>
#include <MemoryAllocator.hpp>

template <typename T>
class CPUObjectCache {
	NUMAObjectCache<T> *_NUMAObjectCache;
	size_t _NUMANodeId;
	size_t _numaNodeCount;
	
	size_t _allocationSize = 1;
	
	typedef std::deque<T *> pool_t;
	
	/** Pools of available objects in the cache.
	 *
	 * We have one such pool per NUMA node, plus one extra for non-NUMA
	 * allocations, e.g. for ExternalThreads. When allocating an object
	 * the local pool will be used. When deleting an object it will be
	 * placed in the pool of the NUMA node in which the underlying memory
	 * belongs to. When the pool of a NUMA node other than ours reaches a
	 * certain limit, the objects of that pool will be returned to the
	 * cache of that NUMA node.
	 *
	 * These pools are not thread safe, i.e. they are meant to be accessed
	 * only by the thread that runs on the current CPU. */
	std::deque<pool_t> _available;
	
public:
	CPUObjectCache(NUMAObjectCache<T> *pool, size_t numaId, size_t numaNodeCount)
		: _NUMAObjectCache(pool), _NUMANodeId(numaId), _numaNodeCount(numaNodeCount)
	{
		_available.resize(numaNodeCount + 1);
	}
	
	~CPUObjectCache()
	{
	}
	
	//! Allocate an object from the current CPU memory pool
	template <typename... TS>
	T *newObject(TS &&... args)
	{
		pool_t &local = _available[_NUMANodeId];
		if (local.empty()) {
			//! Try to recycle from NUMA pool
			_allocationSize *= 2;
			size_t allocated = _NUMAObjectCache->fillCPUPool(_NUMANodeId, local,
					_allocationSize);
			
			//! If NUMA pool did not have objects allocate new memory
			if (allocated == 0) {
				T *ptr = (T *) MemoryAllocator::alloc(
						_allocationSize * sizeof(T));
				for (size_t i = 0; i < _allocationSize; ++i) {
					local.push_back(&ptr[i]);
				}
			}
		}
		
		T *ret = local.front();
		local.pop_front();
		new (ret) T(std::forward<TS>(args)...);
		return ret;
	}
	
	//! Deallocate an object
	void deleteObject(T *ptr)
	{
		size_t nodeId = VirtualMemoryManagement::findNUMA((void *)ptr);
		
		ptr->~T();
		
		_available[nodeId].push_front(ptr);
		if ((nodeId != _NUMANodeId) && (_available[nodeId].size() == 64)) {
			_NUMAObjectCache->returnObjects(nodeId, _available[nodeId]);
		}
	}
};

#endif /* __CPU_OBJECT_CACHE_HPP__ */
