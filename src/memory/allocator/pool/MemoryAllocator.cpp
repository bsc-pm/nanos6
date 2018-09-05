/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/HardwareInfo.hpp"

#include "MemoryPool.hpp"

#include "MemoryAllocator.hpp"

std::vector<MemoryPoolGlobal *> MemoryAllocator::_globalMemoryPool;
std::vector<MemoryAllocator::size_to_pool_t> MemoryAllocator::_localMemoryPool;
MemoryAllocator::size_to_pool_t MemoryAllocator::_externalMemoryPool;
SpinLock MemoryAllocator::_externalMemoryPoolLock;

MemoryPool *MemoryAllocator::getPool(size_t size)
{
	WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
	size_t CPUId;
	size_t NUMANodeId;
	MemoryPool *pool = nullptr;
	
	// Round to the nearest multiple of the cache line size
	size_t cacheLineSize = HardwareInfo::getCacheLineSize();
	size_t roundedSize = (size + cacheLineSize - 1) & ~(cacheLineSize - 1);
	size_t cacheLines = roundedSize / cacheLineSize;
	
	if (thread != nullptr) {
		CPU *currentCPU = thread->getComputePlace();
		
		if (currentCPU != nullptr) {
			CPUId = currentCPU->_virtualCPUId;
			NUMANodeId = currentCPU->_NUMANodeId;
			
			auto it = _localMemoryPool[CPUId].find(cacheLines);
			if (it == _localMemoryPool[CPUId].end()) {
				// No pool of this size locally
				pool = new MemoryPool(_globalMemoryPool[NUMANodeId], roundedSize);
				_localMemoryPool[CPUId][cacheLines] = pool;
			} else {
				pool = it->second;
			}
		}
	}
	
	if (pool == nullptr) {
		std::lock_guard<SpinLock> guard(_externalMemoryPoolLock);
		auto it = _externalMemoryPool.find(cacheLines);
		if (it == _externalMemoryPool.end()) {
			pool = new MemoryPool(_globalMemoryPool[0], roundedSize);
			_externalMemoryPool[cacheLines] = pool;
		} else {
			pool = it->second;
		}
	}
	
	return pool;
}

void MemoryAllocator::initialize()
{
	size_t NUMANodeCount = HardwareInfo::getMemoryNodeCount();
	
	_globalMemoryPool.resize(NUMANodeCount);
	
	for (size_t i = 0; i < NUMANodeCount; ++i) {
		_globalMemoryPool[i] = new MemoryPoolGlobal(i);
	}
	
	_localMemoryPool.resize(HardwareInfo::getComputeNodeCount());
}

void MemoryAllocator::shutdown()
{
	for (size_t i = 0; i < _globalMemoryPool.size(); ++i) {
		delete _globalMemoryPool[i];
	}
	
	for (size_t i = 0; i < _localMemoryPool.size(); ++i) {
		for (auto it = _localMemoryPool[i].begin(); it != _localMemoryPool[i].end(); ++it) {
			delete it->second;
		}
	}
	
	for (auto it = _externalMemoryPool.begin(); it != _externalMemoryPool.end(); ++it) {
		delete it->second;
	}
}

void *MemoryAllocator::alloc(size_t size)
{
	MemoryPool *pool = getPool(size);
	
	return pool->getChunk();
}

void MemoryAllocator::free(void *chunk, size_t size)
{
	MemoryPool *pool = getPool(size);
	
	pool->returnChunk(chunk);
}
