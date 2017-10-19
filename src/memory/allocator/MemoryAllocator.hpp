/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef MEMORY_ALLOCATOR_HPP
#define MEMORY_ALLOCATOR_HPP

#include <map>
#include <vector>

class MemoryPool;
class MemoryPoolGlobal;

class MemoryAllocator {
private:
	typedef std::map<size_t, MemoryPool *> size_to_pool_t;
	
	static SpinLock _lock;
	static std::vector<MemoryPoolGlobal *> _globalMemoryPool;
	static std::vector<size_to_pool_t> _NUMAMemoryPool;
	static std::vector<size_to_pool_t> _localMemoryPool;
	
	static MemoryPool *getPool(size_t size);

public:
	static void initialize();
	static void shutdown();
	
	static void *alloc(size_t size);
	static void free(void *chunk, size_t size);
};

#endif // MEMORY_ALLOCATOR_HPP
