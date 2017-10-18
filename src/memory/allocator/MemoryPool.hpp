/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef MEMORY_POOL_HPP
#define MEMORY_POOL_HPP
#include <atomic>

#include "lowlevel/SpinLock.hpp"
#include "MemoryPoolGlobal.hpp"

#define MAX_CHUNK_REQUEST 20
#define NEXT_CHUNK(_r) *((void **)_r)

class MemoryPool {
private:
	MemoryPoolGlobal *_globalAllocator;
	SpinLock _lock;
	size_t _chunkSize;
	std::atomic<void *> _topChunk;
	
	
	void fillPool()
	{
		size_t globalChunkSize;
		void *allocMemory = _globalAllocator->getMemory(globalChunkSize);
		
		assert(_chunkSize < globalChunkSize);
		
		/* If globalChunkSize % _chunkSize != 0, some memory will be left unused */
		size_t numChunks = globalChunkSize / _chunkSize;
		void *tmpTopChunk = allocMemory;
		void *prevChunk = allocMemory;
		for (size_t i = 1; i < numChunks; ++i) {
			// Link chunks to each other, by writing a pointer to the next chunk in this chunk
			NEXT_CHUNK(prevChunk) = (char *)allocMemory + (i * _chunkSize);
			prevChunk = (char *)allocMemory + (i * _chunkSize);
		}
		
		NEXT_CHUNK(prevChunk) = nullptr;

		// Update the "public" stack, and combine it with any chunks that may have been
		// returned while we were generating a new stack
		void *expected = nullptr;
		while (!_topChunk.compare_exchange_strong(expected, tmpTopChunk))
		{
			NEXT_CHUNK(prevChunk) = expected;
		}
	}

public:
	MemoryPool(MemoryPoolGlobal *globalAllocator, size_t chunkSize)
		: _globalAllocator(globalAllocator),
		_chunkSize(chunkSize),
		_topChunk(nullptr)
	{
		assert(chunkSize > sizeof(void *));
		fillPool();
	}
	
	void *getChunk()
	{
		void *chunk = _topChunk;
		
		// Try to reserve a chunk
		while(chunk != nullptr && !_topChunk.compare_exchange_strong(chunk, NEXT_CHUNK(chunk)));
		
		while(chunk == nullptr) {
			// The pool emptied while we were waiting
			{
				std::lock_guard<SpinLock> guard(_lock);
				// The pool might have been filled while we were waiting for the lock
				if (_topChunk == nullptr) {
					fillPool();
				}
			}
			
			// Try to reserve a chunk
			chunk = _topChunk;
			while(chunk != nullptr && !_topChunk.compare_exchange_strong(chunk, NEXT_CHUNK(chunk)));
		}
		
		return chunk;
	}
	
	void returnChunk(void *chunk)
	{
		NEXT_CHUNK(chunk) = _topChunk;
		while(!_topChunk.compare_exchange_strong(NEXT_CHUNK(chunk), chunk));
	}
};

#endif // MEMORY_POOL_HPP
