/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef MEMORY_ALLOCATOR_HPP
#define MEMORY_ALLOCATOR_HPP

#include <jemalloc/jemalloc-nanos6.h>

#include <cstdint>

#include "lowlevel/FatalErrorHandler.hpp"
#include "lowlevel/Padding.hpp"
#include <InstrumentMemory.hpp>

class MemoryAllocator {
private:
	static const int MALLOCX_NONE = ((int) 0);

public:
	static void initialize()
	{
	}

	static void shutdown()
	{
	}

	static constexpr bool hasUsageStatistics()
	{
		return true;
	}

	static inline size_t getMemoryUsage()
	{
		size_t allocated, size, epsize;
		size = sizeof(allocated);
		uint64_t epoch = 1;
		epsize = sizeof(epoch);

		// Expensive: force a flush of the tcache and an epoch change to refresh the statistics.
		nanos6_je_mallctl("thread.tcache.flush", nullptr, nullptr, nullptr, 0);
		nanos6_je_mallctl("epoch", &epoch, &epsize, &epoch, epsize);

		nanos6_je_mallctl("stats.active", &allocated, &size, nullptr, 0);

		return allocated;
	}

	static inline void *alloc(size_t size)
	{
		assert(size > 0);

		void *ptr = nullptr;
		if (size >= CACHELINE_SIZE / 2) {
			ptr = allocAligned(size);
		} else {
			Instrument::memoryAllocEnter();
			ptr = nanos6_je_mallocx(size, MALLOCX_NONE);
			Instrument::memoryAllocExit();

			if (ptr == nullptr)
				FatalErrorHandler::fail("nanos6_je_mallocx failed to allocate memory");
		}

		return ptr;
	}

	static inline void *allocAligned(size_t size)
	{
		assert(size > 0);

		Instrument::memoryAllocEnter();
		void *ptr = nanos6_je_mallocx(size, MALLOCX_ALIGN(CACHELINE_SIZE));
		Instrument::memoryAllocExit();

		if (ptr == nullptr)
			FatalErrorHandler::fail("nanos6_je_mallocx failed to allocate memory");

		if ((uintptr_t) ptr % CACHELINE_SIZE != 0)
			FatalErrorHandler::fail("nanos6_je_mallocx failed to allocate cache aligned memory");

		return ptr;
	}

	static inline void free(void *chunk, size_t size)
	{
		assert(size > 0);

		if (size >= CACHELINE_SIZE / 2) {
			freeAligned(chunk, size);
		} else {
			Instrument::memoryFreeEnter();
			nanos6_je_sdallocx(chunk, size, MALLOCX_NONE);
			Instrument::memoryFreeExit();
		}
	}

	static inline void freeAligned(void *chunk, size_t size)
	{
		assert(size > 0);

		Instrument::memoryFreeEnter();
		nanos6_je_sdallocx(chunk, size, MALLOCX_ALIGN(CACHELINE_SIZE));
		Instrument::memoryFreeExit();
	}

	// Simplifications for using "new" and "delete" with the allocator
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

template <class T>
class TemplateAllocator {
public:
	using value_type = T;
	using reference = T &;
	using const_reference = T const&;
	using pointer = T *;
	using const_pointer = T const*;
	using size_type = std::size_t;
	using difference_type = std::ptrdiff_t;

	TemplateAllocator() = default;
	TemplateAllocator(const TemplateAllocator &) = default;

	inline T *allocate(size_t n, const void* = nullptr)
	{
		return static_cast<T*>(MemoryAllocator::alloc(n * sizeof(T)));
	}

	inline void deallocate(void *t, size_t size)
	{
		assert(t != nullptr);
		MemoryAllocator::free(t, size * sizeof(T));
	}

	template<class U>
	TemplateAllocator(const TemplateAllocator<U>&) {}

	// Legacy C++98/C++03 hack for rebinding operators that is needed for
	// repurposing allocator instances in older Boost versions.
	template <class U>
	struct rebind {typedef TemplateAllocator<U> other;};

	template <class U, class ...Args>
	void construct(U* p, Args&& ...args) 
	{
		::new(p) U(std::forward<Args>(args)...);
	}

	template <class U>
	void destroy(U* p) noexcept
	{
		p->~U();
	}
};

template <class T, class U>
inline bool operator==(TemplateAllocator<T> const&, TemplateAllocator<U> const&) noexcept
{
    return true;
}

template <class T, class U>
inline bool operator!=(TemplateAllocator<T> const& x, TemplateAllocator<U> const& y) noexcept
{
    return !(x == y);
}

#endif // MEMORY_ALLOCATOR_HPP
