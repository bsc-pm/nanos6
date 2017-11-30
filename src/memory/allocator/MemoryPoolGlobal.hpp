/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef MEMORY_POOL_GLOBAL_HPP
#define MEMORY_POOL_GLOBAL_HPP
#include <vector>

#if HAVE_MEMKIND
#include <memkind.h>
#endif

#include "lowlevel/SpinLock.hpp"
#include "lowlevel/EnvironmentVariable.hpp"

class MemoryPoolGlobal {
private:
	size_t _globalAllocSize;
	size_t _memoryChunkSize;

	SpinLock _lock;
	size_t _pageSize;
	std::vector<void *> _oldMemoryChunks;
	void *_curMemoryChunk;
	size_t _curAvailable;
	size_t _NUMANodeId;
#if HAVE_MEMKIND
	memkind_t _memoryKind;
#endif

	void fillPool()
	{
		assert(_curAvailable == 0);
		_curAvailable = _globalAllocSize;
#if HAVE_MEMKIND
		int rc = memkind_posix_memalign(_memoryKind, &_curMemoryChunk, _pageSize, _globalAllocSize);
		FatalErrorHandler::check(rc == MEMKIND_SUCCESS, " when trying to allocate a memory chunk for the global allocator");
#else
		int rc = posix_memalign(&_curMemoryChunk, _pageSize, _globalAllocSize);
		FatalErrorHandler::handle(rc, " when trying to allocate a memory chunk for the global allocator");
#endif
		_oldMemoryChunks.push_back(_curMemoryChunk);
	}

public:
	MemoryPoolGlobal(size_t NUMANodeId)
		: _pageSize(sysconf(_SC_PAGESIZE)), _oldMemoryChunks(0),
		_curMemoryChunk(nullptr), _curAvailable(0),
		_NUMANodeId(NUMANodeId)
	{
		EnvironmentVariable<size_t> globalAllocSize("NANOS6_GLOBAL_ALLOC_SIZE", 64 * 1024 * 1024);
		_globalAllocSize = globalAllocSize;
		
		EnvironmentVariable<size_t> memoryChunkSize("NANOS6_ALLOCATOR_CHUNK_SIZE", 64 * 1024);
		_memoryChunkSize = memoryChunkSize;
		
#if HAVE_MEMKIND
		int rc = memkind_create_kind(MEMKIND_MEMTYPE_DEFAULT, MEMKIND_POLICY_PREFERRED_LOCAL, (memkind_bits_t)0, &_memoryKind);
		FatalErrorHandler::check(rc == MEMKIND_SUCCESS, " when trying to create a new memory kind");
#endif

		fillPool();
	}
	
	~MemoryPoolGlobal()
	{
		for (auto it : _oldMemoryChunks) {
#if HAVE_MEMKIND
			memkind_free(_memoryKind, it);
#else
			free(it);
#endif
		}
	}
	
	void *getMemory(size_t &chunkSize)
	{
		std::lock_guard<SpinLock> guard(_lock);
		if (_curAvailable == 0) {
			fillPool();
		}
		
		void *curAddr = _curMemoryChunk;
		
		chunkSize = _memoryChunkSize;
		_curAvailable -= chunkSize;
		_curMemoryChunk = (char *)_curMemoryChunk + chunkSize;
		
		return curAddr;
	}
};

#endif // MEMORY_POOL_GLOBAL_HPP
