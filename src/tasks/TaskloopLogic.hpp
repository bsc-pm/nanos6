/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKLOOP_LOGIC_HPP
#define TASKLOOP_LOGIC_HPP

#include <nanos6.h>

#include <vector>

class TaskloopLogic {
private:
	typedef nanos6_taskloop_bounds_t bounds_t;
	
public:
	static inline void splitIterations(int numPartitions, const bounds_t &bounds, std::vector<bounds_t> &partitions)
	{
		assert(numPartitions > 0);
		
		// The loop must be normalized
		assert(bounds.step == 1);
		const size_t originalLowerBound = bounds.lower_bound;
		const size_t originalUpperBound = bounds.upper_bound;
		const size_t chunksize = bounds.chunksize;
		
		// Compute the number of chunks per partition
		const size_t totalIterations = getIterationCount(bounds);
		const size_t totalChunks = (totalIterations / chunksize) + (totalIterations % chunksize > 0);
		const size_t chunksPerPartition = (totalChunks / numPartitions);
		
		bounds_t partialBounds;
		partialBounds.step = 1;
		partialBounds.chunksize = chunksize;
		
		// Start the partition from the original lower bound
		size_t lowerBound = originalLowerBound;
		
		for (int partition = 0; partition < numPartitions; ++partition) {
			// Compute the upper bound for this partition
			const size_t extraChunk = (totalChunks % numPartitions >= (size_t) numPartitions - partition);
			const size_t myIterations = (chunksPerPartition + extraChunk) * chunksize;
			const size_t upperBound = std::min(lowerBound + myIterations, originalUpperBound);
			
			partialBounds.lower_bound = lowerBound;
			partialBounds.upper_bound = upperBound;
			
			// Set taskloop bounds
			partitions.push_back(partialBounds);
			
			// Update the lower bound for the next partition
			lowerBound = upperBound;
		}
	}
	
	static inline size_t getIterationCount(const bounds_t &bounds)
	{
		return (bounds.upper_bound - bounds.lower_bound);
	}
};

#endif
