/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKLOOP_INFO_HPP
#define TASKLOOP_INFO_HPP

#include <atomic>
#include <algorithm>

#include <nanos6.h>
#include "executors/threads/CPUManager.hpp"

#define CACHE_LINE_SIZE 128

#ifndef ALIGNED
#define ALIGNED __attribute__ ((aligned (CACHE_LINE_SIZE)))
#endif

struct TaskloopInfo {
	typedef nanos6_loop_bounds_t bounds_t;
private:
	
	friend class Taskloop;
	
protected:
	bounds_t _bounds;
	bool _sourceTaskloop;
	
public:
	inline TaskloopInfo()
		: _bounds(), _sourceTaskloop(false)
	{}

	inline void reinitialize() 
	{
		_bounds = bounds_t();
	}
	
	inline ~TaskloopInfo()
	{}
	
	inline void initialize()
	{
		size_t totalIterations = getIterationCount();

		// Set a implementation defined chunksize if needed
		if (_bounds.grainsize == 0) {
			_bounds.grainsize = std::max(totalIterations /CPUManager::getTotalCPUs(), (size_t) 1);
		}
	}
	
	inline void initialize(size_t lowerBound, size_t upperBound, size_t grainsize)
	{
		_bounds.lower_bound = lowerBound;
		_bounds.upper_bound = upperBound;
		_bounds.grainsize = grainsize;
		_sourceTaskloop = true;
		
		initialize();
	}
	
	inline void initialize(const bounds_t &newBounds)
	{
		// Set the bounds
		_bounds.lower_bound = newBounds.lower_bound;
		_bounds.upper_bound = newBounds.upper_bound;
		_bounds.grainsize = newBounds.grainsize;
		_bounds.chunksize = newBounds.chunksize;
		
		initialize();
	}
	
	inline bounds_t &getBounds()
	{
		return _bounds;
	}
	
	inline bounds_t const &getBounds() const
	{
		return _bounds;
	}

	inline void setBounds(bounds_t &bounds) 
	{
		_bounds.lower_bound = bounds.lower_bound;
		_bounds.upper_bound = bounds.upper_bound;
	}

	inline size_t getIterationCount()
	{
		return (_bounds.upper_bound - _bounds.lower_bound);
	}

	inline bool isSourceTaskloop()
	{
		return _sourceTaskloop;
	}
};

#endif // TASKLOOP_INFO_HPP
