/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/HardwareInfo.hpp"
#include <VirtualMemoryManagement.hpp>

#include "MemoryPool.hpp"

#include "MemoryAllocator.hpp"
#include "ObjectAllocator.hpp"

std::vector<MemoryPoolGlobal *> MemoryAllocator::_globalMemoryPool;
std::vector<MemoryAllocator::size_to_pool_t> MemoryAllocator::_localMemoryPool;
MemoryAllocator::size_to_pool_t MemoryAllocator::_externalMemoryPool;
SpinLock MemoryAllocator::_externalMemoryPoolLock;

MemoryPool *MemoryAllocator::getPool(size_t size)
{
	WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
	size_t cpuId;
	size_t numaNodeId;
	MemoryPool *pool = nullptr;
	
	// Round to the nearest multiple of the cache line size
	size_t cacheLineSize = HardwareInfo::getCacheLineSize();
	size_t roundedSize = (size + cacheLineSize - 1) & ~(cacheLineSize - 1);
	size_t cacheLines = roundedSize / cacheLineSize;
	
	if (thread != nullptr) {
		CPU *currentCPU = thread->getComputePlace();
		
		if (currentCPU != nullptr) {
			cpuId = currentCPU->_virtualCPUId;
			numaNodeId = currentCPU->_NUMANodeId;
			
			auto it = _localMemoryPool[cpuId].find(cacheLines);
			if (it == _localMemoryPool[cpuId].end()) {
				// No pool of this size locally
				pool = new MemoryPool(_globalMemoryPool[numaNodeId], roundedSize);
				_localMemoryPool[cpuId][cacheLines] = pool;
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
	VirtualMemoryManagement::initialize();
	
	size_t numaNodeCount = HardwareInfo::getMemoryPlaceCount(nanos6_device_t::nanos6_host_device);
	size_t cpuCount = HardwareInfo::getComputePlaceCount(nanos6_device_t::nanos6_host_device);	
	_globalMemoryPool.resize(numaNodeCount);
	
	for (size_t i = 0; i < numaNodeCount; ++i) {
		_globalMemoryPool[i] = new MemoryPoolGlobal(i);
	}
	
	_localMemoryPool.resize(cpuCount);
	
	//! Initialize the Object caches
	ObjectAllocator<DataAccess>::initialize();
	ObjectAllocator<ReductionInfo>::initialize();
	ObjectAllocator<BottomMapEntry>::initialize();
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
	
	//! Initialize the Object caches
	ObjectAllocator<BottomMapEntry>::shutdown();
	ObjectAllocator<ReductionInfo>::shutdown();
	ObjectAllocator<DataAccess>::shutdown();
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
