/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef MEMORY_POOL_GLOBAL_HPP
#define MEMORY_POOL_GLOBAL_HPP
#include <vector>

#include "lowlevel/SpinLock.hpp"

#define GLOBAL_ALLOC_SIZE (16*1024*1024)
#define GLOBAL_ALIGNMENT 1024
#define MEMORY_CHUNK_SIZE (1*1024*1024)

class MemoryPoolGlobal {
private:
	SpinLock _lock;
	std::vector<void *> _oldMemoryChunks;
	void *_curMemoryChunk;
	size_t _curAvailable;
	size_t _NUMANodeId;

	void fillPool()
	{
		assert(_curAvailable == 0);
		_curAvailable = GLOBAL_ALLOC_SIZE;
		// TODO: alloc on an specific NUMA node
		int rc = posix_memalign(&_curMemoryChunk, GLOBAL_ALIGNMENT, GLOBAL_ALLOC_SIZE);
		FatalErrorHandler::handle(rc, " when trying to allocate a memory chunk for the global allocator");
		//_curMemoryChunk = malloc(_curAvailable);
		_oldMemoryChunks.push_back(_curMemoryChunk);
	}

public:
	MemoryPoolGlobal(size_t NUMANodeId)
		: _oldMemoryChunks(0), _curMemoryChunk(nullptr),
		_curAvailable(0), _NUMANodeId(NUMANodeId)
	{
		fillPool();
	}
	
	~MemoryPoolGlobal()
	{
		for (auto it : _oldMemoryChunks) {
			// TODO: free from an specific NUMA node
			free(it);
		}
	}
	
	void *getMemory(size_t &chunkSize)
	{
		std::lock_guard<SpinLock> guard(_lock);
		if (_curAvailable == 0) {
			fillPool();
		}
		
		void *curAddr = _curMemoryChunk;
		
		chunkSize = MEMORY_CHUNK_SIZE;
		_curAvailable -= chunkSize;
		_curMemoryChunk = (char *)_curMemoryChunk + chunkSize;
		
		return curAddr;
	}
};

#endif // MEMORY_POOL_GLOBAL_HPP
