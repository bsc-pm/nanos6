/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef EXTERNAL_MEMORY_POOL_HPP
#define EXTERNAL_MEMORY_POOL_HPP

#include "MemoryPool.hpp"
#include "MemoryPoolGlobal.hpp"

class ExternalMemoryPool : public MemoryPool {
private:
	SpinLock _lock;

public:
	ExternalMemoryPool(MemoryPoolGlobal *globalAllocator, size_t chunkSize)
		: MemoryPool(globalAllocator, chunkSize);
	{
	}
	
	void *getChunk()
	{
		std::lock_guard<SpinLock> guard(_lock);
		return MemoryPool::getChunk();
	}
	
	void returnChunk(void *chunk)
	{
		std::lock_guard<SpinLock> guard(_lock);
		MemoryPool::returnChunk(chunk);
	}
};

#endif // EXTERNAL_MEMORY_POOL_HPP
