/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef MEMORY_ALLOCATOR_HPP
#define MEMORY_ALLOCATOR_HPP

#include <map>
#include <vector>

#include "lowlevel/SpinLock.hpp"

class MemoryPool;
class MemoryPoolGlobal;
class Task;
class DataAccess;

class MemoryAllocator {
private:
	typedef std::map<size_t, MemoryPool *> size_to_pool_t;
	
	static std::vector<MemoryPoolGlobal *> _globalMemoryPool;
	static std::vector<size_to_pool_t> _localMemoryPool;
	
	static size_to_pool_t _externalMemoryPool;
	static SpinLock _externalMemoryPoolLock;
	
	static MemoryPool *getPool(size_t size);
	
public:
	static void initialize();
	static void shutdown();
	
	static void *alloc(size_t size);
	static void free(void *chunk, size_t size);
	
	/* Simplifications for using "new" and "delete" with the allocator */
	template <typename T, typename... Args>
	static T *newObject(Args &&... args)
	{
		void *ptr = MemoryAllocator::alloc(sizeof(T));
		new (ptr) T(std::forward<Args>(args)...);
		return (T*)ptr;
	}
	
	template <typename T>
	static void deleteObject(T *ptr)
	{
		ptr->~T();
		MemoryAllocator::free(ptr, sizeof(T));
	}
};

#endif // MEMORY_ALLOCATOR_HPP
