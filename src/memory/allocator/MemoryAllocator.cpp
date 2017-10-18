/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <assert.h>
#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/HardwareInfo.hpp"

#include "MemoryPool.hpp"

#include "MemoryAllocator.hpp"

#define CACHE_LINE_SIZE 64

SpinLock MemoryAllocator::_lock;
std::vector<MemoryPoolGlobal *> MemoryAllocator::_globalMemoryPool;
std::vector<MemoryAllocator::size_to_pool_t> MemoryAllocator::_NUMAMemoryPool;
std::vector<MemoryAllocator::size_to_pool_t> MemoryAllocator::_localMemoryPool;

MemoryPool *MemoryAllocator::getPool(size_t cacheLines)
{
	WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
	size_t CPUId;
	size_t NUMANodeId;
	
	if (thread != nullptr) {
		CPU *currentCPU = thread->getComputePlace();
		CPUId = currentCPU->_virtualCPUId;
		NUMANodeId = currentCPU->_NUMANodeId;
	} else {
		CPUId = 0;
		NUMANodeId = 0;
	}
	
	MemoryPool *pool = nullptr;
	auto it = _localMemoryPool[CPUId].find(cacheLines);
	if (it == _localMemoryPool[CPUId].end()) {
		// No pool of this size locally
		std::lock_guard<SpinLock> guard(_lock);
		auto itNUMA = _NUMAMemoryPool[NUMANodeId].find(cacheLines);
		if (itNUMA == _NUMAMemoryPool[NUMANodeId].end()) {
			// No pool of this size in the NUMA node
			pool = new MemoryPool(_globalMemoryPool[NUMANodeId], cacheLines * CACHE_LINE_SIZE);
			_NUMAMemoryPool[NUMANodeId][cacheLines] = pool;
		} else {
			pool = itNUMA->second;
		}
	
		_localMemoryPool[CPUId][cacheLines] = pool;
	} else {
		pool = it->second;
	}

	assert(pool != nullptr);
	return pool;
}

void MemoryAllocator::initialize()
{
	size_t NUMANodeCount = HardwareInfo::getMemoryNodeCount();
	
	_globalMemoryPool.resize(NUMANodeCount);
	
	for (size_t i = 0; i < NUMANodeCount; ++i) {
		_globalMemoryPool[i] = new MemoryPoolGlobal(i);
	}
	
	_NUMAMemoryPool.resize(NUMANodeCount);
	_localMemoryPool.resize(HardwareInfo::getComputeNodeCount());
}

void MemoryAllocator::shutdown()
{
	// TODO: delete other structures
	
	for (size_t i = 0; i < _globalMemoryPool.size(); ++i) {
		delete _globalMemoryPool[i];
	}
}

void *MemoryAllocator::alloc(size_t size)
{
	size_t cacheLines = size / CACHE_LINE_SIZE;
	
	if (size % CACHE_LINE_SIZE != 0) {
		cacheLines += 1;
	}
	
	MemoryPool *pool = getPool(cacheLines);
	
	return pool->getChunk();
}

void MemoryAllocator::free(void *chunk, size_t size)
{
	size_t cacheLines = size / CACHE_LINE_SIZE;
	
	if (size % CACHE_LINE_SIZE != 0) {
		cacheLines += 1;
	}
	
	MemoryPool *pool = getPool(cacheLines);
	
	pool->returnChunk(chunk);
}
