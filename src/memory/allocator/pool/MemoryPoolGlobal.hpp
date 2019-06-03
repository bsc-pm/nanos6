/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef MEMORY_POOL_GLOBAL_HPP
#define MEMORY_POOL_GLOBAL_HPP
#include <vector>

#if HAVE_CONFIG_H
#include <config.h>
#endif

#if HAVE_MEMKIND
#include <memkind.h>
#endif

#include <numa.h>

#include "lowlevel/SpinLock.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include <VirtualMemoryManagement.hpp>

class MemoryPoolGlobal {
private:
	EnvironmentVariable<size_t> _globalAllocSize;
	EnvironmentVariable<size_t> _memoryChunkSize;

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
		_curAvailable = _globalAllocSize;
#if HAVE_MEMKIND
		int rc = memkind_posix_memalign(_memoryKind, &_curMemoryChunk, _pageSize, _globalAllocSize);
		FatalErrorHandler::check(rc == MEMKIND_SUCCESS, " when trying to allocate a memory chunk for the global allocator");
#else
		_curMemoryChunk = VirtualMemoryManagement::allocLocalNUMA(_globalAllocSize, _NUMANodeId);
		FatalErrorHandler::failIf(
			_curMemoryChunk == nullptr,
			"could not allocate a memory chunk for the global allocator"
		);

#endif
		if (numa_available() != -1) {
			numa_setlocal_memory(_curMemoryChunk, _globalAllocSize);
		}
		
		_oldMemoryChunks.push_back(_curMemoryChunk);
	}

public:
	MemoryPoolGlobal(size_t NUMANodeId)
		: _globalAllocSize("NANOS6_GLOBAL_ALLOC_SIZE", 8 * 1024 * 1024),
		_memoryChunkSize("NANOS6_ALLOCATOR_CHUNK_SIZE", 128 * 1024),
		_pageSize(sysconf(_SC_PAGESIZE)), _oldMemoryChunks(0),
		_curMemoryChunk(nullptr), _curAvailable(0),
		_NUMANodeId(NUMANodeId)
	{
		FatalErrorHandler::failIf((_globalAllocSize % _memoryChunkSize) != 0, "Pool size and chunk size must be multiples of eachother");
#if HAVE_MEMKIND
		int rc = memkind_create_kind(MEMKIND_MEMTYPE_DEFAULT, MEMKIND_POLICY_PREFERRED_LOCAL, (memkind_bits_t)0, &_memoryKind);
		FatalErrorHandler::check(rc == MEMKIND_SUCCESS, " when trying to create a new memory kind");
#endif

		fillPool();
	}
	
	~MemoryPoolGlobal()
	{
#if HAVE_MEMKIND
		for (auto it : _oldMemoryChunks) {
			memkind_free(_memoryKind, it);
		}
#endif
	}
	
	void *getMemory(size_t minSize, size_t &chunkSize)
	{
		std::lock_guard<SpinLock> guard(_lock);
		if (_curAvailable < _memoryChunkSize) {
			if (_curAvailable != 0) {
				// Chunk size was changed previously, update also alloc size to make all sizes fit again
				_globalAllocSize.setValue(((_globalAllocSize + _memoryChunkSize - 1) / _memoryChunkSize) * _memoryChunkSize);
			}
			
			fillPool();
		}
		
		chunkSize = _memoryChunkSize;
		
		if (chunkSize < minSize) {
			// Get minimum acceptable chunkSize
			chunkSize = ((minSize + _memoryChunkSize - 1) / _memoryChunkSize) * _memoryChunkSize;
			_memoryChunkSize.setValue(chunkSize);
			
			if (_curAvailable < chunkSize) {
				_globalAllocSize.setValue(((_globalAllocSize + _memoryChunkSize - 1) / _memoryChunkSize) * _memoryChunkSize);
				fillPool();
			}
		}
		
		void *curAddr = _curMemoryChunk;
		_curAvailable -= chunkSize;
		_curMemoryChunk = (char *)_curMemoryChunk + chunkSize;
		
		return curAddr;
	}
};

#endif // MEMORY_POOL_GLOBAL_HPP
