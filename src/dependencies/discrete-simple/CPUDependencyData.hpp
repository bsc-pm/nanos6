/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_DEPENDENCY_DATA_HPP
#define CPU_DEPENDENCY_DATA_HPP


#include <atomic>
#include <cassert>
#include <deque>

#include <limits.h>

class Task;


struct CPUDependencyData {
	typedef std::deque<Task *> satisfied_originator_list_t;
	
	//! Tasks whose accesses have been satisfied after ending a task
	satisfied_originator_list_t _satisfiedOriginators;
	
#ifndef NDEBUG
	std::atomic<bool> _inUse;
#endif
	
	CPUDependencyData()
		: _satisfiedOriginators()
#ifndef NDEBUG
		, _inUse(false)
#endif
	{
	}
	
	~CPUDependencyData()
	{
		assert(empty());
	}
	
	inline bool empty() const
	{
		return _satisfiedOriginators.empty();
	}
};


#endif // CPU_DEPENDENCY_DATA_HPP
