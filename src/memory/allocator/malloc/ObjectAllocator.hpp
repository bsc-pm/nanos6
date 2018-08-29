/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef __OBJECT_ALLOCATOR_HPP__
#define __OBJECT_ALLOCATOR_HPP__

#include <DataAccess.hpp>
#include <ReductionInfo.hpp>
#include <BottomMapEntry.hpp>
#include <MemoryAllocator.hpp>

template<typename T>
class ObjectAllocator {
public:
	template<typename... ARGS>
	static inline T *newObject(ARGS &&... args)
	{
		return MemoryAllocator::newObject<T>(
				std::forward<ARGS>(args)...);
	}
	
	static inline void deleteObject(T *ptr)
	{
		MemoryAllocator::deleteObject<T>(ptr);
	}
};

#endif /* __OBJECT_ALLOCATOR_HPP__ */
