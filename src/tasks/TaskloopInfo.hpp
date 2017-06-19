#ifndef TASKLOOP_INFO_HPP
#define TASKLOOP_INFO_HPP

#include <atomic>

#include <nanos6.h>
#include "tasks/TaskloopBounds.hpp"
#include "lowlevel/SpinLock.hpp"

#define CACHE_LINE_SIZE 128

struct TaskloopInfo {
	std::atomic<size_t> _nextLowerBound __attribute__ ((aligned (CACHE_LINE_SIZE)));
	
	nanos6_taskloop_bounds_t _bounds __attribute__ ((aligned (CACHE_LINE_SIZE)));
	
	inline TaskloopInfo()
		: _nextLowerBound(0),
		_bounds()
	{
	}
	
	inline void initialize()
	{
		_nextLowerBound = _bounds.lower_bound;
	}
	
	inline void setBounds(size_t lowerBound, size_t upperBound, size_t chunksize, size_t step)
	{
		_bounds.lower_bound = lowerBound;
		_bounds.upper_bound = upperBound;
		_bounds.chunksize = chunksize;
		_bounds.step = step;
		
		initialize();
	}
	
	inline void setBounds(const nanos6_taskloop_bounds_t &newBounds)
	{
		setBounds(
			newBounds.lower_bound,
			newBounds.upper_bound,
			newBounds.chunksize,
			newBounds.step
		);
	}
};

#endif // TASKLOOP_INFO_HPP
