/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef MEMORY_POOL_HPP
#define MEMORY_POOL_HPP

#include "MemoryPoolGlobal.hpp"

#define NEXT_CHUNK(_r) *((void **)_r)

class MemoryPool {
private:
	// There is one pool per CPU. No need to lock
	MemoryPoolGlobal *_globalAllocator;
	size_t _chunkSize;
	void *_topChunk;
	
	
	void fillPool()
	{
		size_t globalChunkSize;
		_topChunk = _globalAllocator->getMemory(_chunkSize, globalChunkSize);
		
		// If globalChunkSize % _chunkSize != 0, some memory will be left unused
		size_t numChunks = globalChunkSize / _chunkSize;
		
		FatalErrorHandler::failIf(numChunks == 0, "Memory returned from global pool is smaller than chunk size (", _chunkSize, "B)");
		
		void *prevChunk = _topChunk;
		for (size_t i = 1; i < numChunks; ++i) {
			// Link chunks to each other, by writing a pointer to the next chunk in this chunk
			NEXT_CHUNK(prevChunk) = (char *)_topChunk + (i * _chunkSize);
			prevChunk = (char *)_topChunk + (i * _chunkSize);
		}
		
		NEXT_CHUNK(prevChunk) = nullptr;
	}

public:
	MemoryPool(MemoryPoolGlobal *globalAllocator, size_t chunkSize)
		: _globalAllocator(globalAllocator),
		_chunkSize(chunkSize),
		_topChunk(nullptr)
	{
	}
	
	void *getChunk()
	{
		if(_topChunk == nullptr) {
			fillPool();
		}
		
		void *chunk = _topChunk;
		_topChunk = NEXT_CHUNK(chunk);
		
		return chunk;
	}
	
	void returnChunk(void *chunk)
	{
		NEXT_CHUNK(chunk) = _topChunk;
		_topChunk = chunk;
	}
};

#endif // MEMORY_POOL_HPP
