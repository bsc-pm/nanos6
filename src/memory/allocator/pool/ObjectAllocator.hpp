/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef __OBJECT_ALLOCATOR_HPP__
#define __OBJECT_ALLOCATOR_HPP__

#include <DataAccess.hpp>
#include <ReductionInfo.hpp>
#include <BottomMapEntry.hpp>
#include <ObjectCache.hpp>

template <typename T>
class ObjectAllocator {
public:
	typedef ObjectCache<T> inner_type;
	
private:
	static inner_type *_cache;
	
public:
	static void initialize()
	{
		_cache = new ObjectCache<T>();
	}
	
	static void shutdown()
	{
		delete _cache;
	}
	
	template <typename... ARGS>
	static inline T *newObject(ARGS &&... args)
	{
		return _cache->newObject(std::forward<ARGS>(args)...);
	}
	
	static inline void deleteObject(T *ptr)
	{
		_cache->deleteObject(ptr);
	}
};


template<> ObjectAllocator<DataAccess>::inner_type *ObjectAllocator<DataAccess>::_cache;
template<> ObjectAllocator<ReductionInfo>::inner_type *ObjectAllocator<ReductionInfo>::_cache;
template<> ObjectAllocator<BottomMapEntry>::inner_type *ObjectAllocator<BottomMapEntry>::_cache;

#endif /* __OBJECT_ALLOCATOR_HPP__ */
