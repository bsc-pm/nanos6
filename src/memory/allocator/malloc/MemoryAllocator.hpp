/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef MEMORY_ALLOCATOR_HPP
#define MEMORY_ALLOCATOR_HPP

#include <cstdint>
#include <cstdlib>
#include <malloc.h>
#include <memory>

#include "lowlevel/FatalErrorHandler.hpp"
#include "lowlevel/Padding.hpp"
#include "InstrumentMemory.hpp"

class MemoryAllocator {
public:
	static inline void initialize()
	{
	}

	static inline void shutdown()
	{
	}

	static constexpr bool hasUsageStatistics()
	{
		return false;
	}

	static inline size_t getMemoryUsage()
	{
		return 0;
	}

	static inline void *alloc(size_t size)
	{
		void *ptr = nullptr;
		Instrument::memoryAllocEnter();

		if (size >= CACHELINE_SIZE / 2) {
			ptr = allocAligned(size);
		} else {
			ptr = malloc(size);
			if (ptr == nullptr)
				FatalErrorHandler::fail("malloc failed to allocate memory");
		}

		Instrument::memoryAllocExit();
		return ptr;
	}

	static inline void *allocAligned(size_t size)
	{
		void *ptr = nullptr;

		int rc = posix_memalign(&ptr, CACHELINE_SIZE, size);
		FatalErrorHandler::handle(rc, " when allocating with posix_memalign");

		if ((uintptr_t) ptr % CACHELINE_SIZE != 0)
			FatalErrorHandler::fail("posix_memalign failed to allocate cache aligned memory");

		return ptr;
	}

	static inline void free(void *chunk, __attribute__((unused)) size_t size)
	{
		Instrument::memoryFreeEnter();
		std::free(chunk);
		Instrument::memoryFreeExit();
	}

	static inline void freeAligned(void *chunk, size_t size)
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

template<typename T>
using TemplateAllocator = std::allocator<T>;

#endif // MEMORY_ALLOCATOR_HPP
