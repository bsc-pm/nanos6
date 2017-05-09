#ifndef TASKLOOP_INFO_HPP
#define TASKLOOP_INFO_HPP

#include <atomic>

#include <nanos6.h>
#include "tasks/TaskloopBounds.hpp"
#include "lowlevel/SpinLock.hpp"

struct TaskloopInfo {
	
	std::atomic<size_t> _remainingIterations;
	std::atomic<size_t> _nextLowerBound;
	
	nanos_taskloop_bounds *_bounds;
	
	inline TaskloopInfo(nanos_taskloop_bounds *bounds)
		: _remainingIterations(0),
		_nextLowerBound(0),
		_bounds(bounds)
	{
	}
	
	inline void initialize()
	{
		assert(_bounds != nullptr);
		
		_remainingIterations = TaskloopBounds::getIterationCount(_bounds);
		_nextLowerBound = _bounds->lower_bound;
	}
	
	inline void setBounds(size_t lowerBound, size_t upperBound, size_t gridSize, size_t step)
	{
		assert(_bounds != nullptr);
		
		_bounds->lower_bound = lowerBound;
		_bounds->upper_bound = upperBound;
		_bounds->grid_size = gridSize;
		_bounds->step = step;
		
		initialize();
	}
	
	inline void setBounds(const nanos_taskloop_bounds &newBounds)
	{
		setBounds(
			newBounds.lower_bound,
			newBounds.upper_bound,
			newBounds.grid_size,
			newBounds.step
		);
	}
};

#endif // TASKLOOP_INFO_HPP
