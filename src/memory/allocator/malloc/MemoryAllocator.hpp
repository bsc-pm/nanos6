/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef MEMORY_ALLOCATOR_HPP
#define MEMORY_ALLOCATOR_HPP

#include <stdlib.h>

#include "hardware/HardwareInfo.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

class MemoryAllocator {
public:
	static inline void initialize()
	{
	}
	
	static inline void shutdown()
	{
	}
	
	static inline void *alloc(size_t size)
	{
		static size_t cacheLineSize = HardwareInfo::getCacheLineSize();
		
		void *ptr;
		
		if (size < cacheLineSize / 2) {
			ptr = malloc(size);
			FatalErrorHandler::failIf(ptr == nullptr, " when trying to allocate memory");
		} else {
			int rc = posix_memalign(&ptr, cacheLineSize, size);
			FatalErrorHandler::handle(rc, " when trying to allocate memory");
		}
		
		return ptr;
	}
	
	static inline void free(void *chunk, __attribute__((unused)) size_t size)
	{
		std::free(chunk);
	}
	
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
