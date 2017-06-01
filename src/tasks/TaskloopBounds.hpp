#ifndef TASKLOOP_BOUNDS_HPP
#define TASKLOOP_BOUNDS_HPP

#include <nanos6.h>

class TaskloopBounds {
public:
	static inline size_t getIterationCount(const nanos_taskloop_bounds &bounds)
	{
		return 1 + ((getRawSize(bounds) - 1) / bounds.step);
	}
	
	static inline size_t getRawSize(const nanos_taskloop_bounds &bounds)
	{
		return bounds.upper_bound - bounds.lower_bound;
	}
};

#endif // TASKLOOP_BOUNDS_HPP
