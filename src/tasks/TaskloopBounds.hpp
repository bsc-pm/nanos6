#ifndef TASKLOOP_BOUNDS_HPP
#define TASKLOOP_BOUNDS_HPP

#include <nanos6.h>

class TaskloopBounds {
public:
	static inline size_t getIterationCount(nanos_taskloop_bounds *bounds)
	{
		assert(bounds != nullptr);
		
		return 1 + ((getRawSize(bounds) - 1) / bounds->step);
	}
	
	static inline size_t getRawSize(nanos_taskloop_bounds *bounds)
	{
		assert(bounds != nullptr);
		
		return bounds->upper_bound - bounds->lower_bound;
	}
};

#endif // TASKLOOP_BOUNDS_HPP
