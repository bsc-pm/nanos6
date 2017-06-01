#ifndef TASKLOOP_INFO_HPP
#define TASKLOOP_INFO_HPP

#include <atomic>

#include <nanos6.h>
#include "tasks/TaskloopBounds.hpp"
#include "lowlevel/SpinLock.hpp"

#define CACHE_LINE_SIZE 128

struct TaskloopInfo {
	std::atomic<size_t> _nextLowerBound __attribute__ ((aligned (CACHE_LINE_SIZE)));
	
	nanos_taskloop_bounds _bounds __attribute__ ((aligned (CACHE_LINE_SIZE)));
	
	inline TaskloopInfo()
		: _nextLowerBound(0),
		_bounds()
	{
	}
	
	inline void initialize()
	{
		_nextLowerBound = _bounds.lower_bound;
	}
	
	inline void setBounds(size_t lowerBound, size_t upperBound, size_t grainSize, size_t step)
	{
		_bounds.lower_bound = lowerBound;
		_bounds.upper_bound = upperBound;
		_bounds.grain_size = grainSize;
		_bounds.step = step;
		
		initialize();
	}
	
	inline void setBounds(const nanos_taskloop_bounds &newBounds)
	{
		setBounds(
			newBounds.lower_bound,
			newBounds.upper_bound,
			newBounds.grain_size,
			newBounds.step
		);
	}
};

#endif // TASKLOOP_INFO_HPP
