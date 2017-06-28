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
		
		const size_t originalLowerBound = bounds.lower_bound;
		const size_t originalUpperBound = bounds.upper_bound;
		const size_t chunksize = bounds.chunksize;
		const size_t step = bounds.step;
		
		// Compute the actual number of iterations
		const size_t totalIts = getIterationCount(bounds);
		size_t itsPerPartition = totalIts / numPartitions;
		const size_t missalignment = itsPerPartition % chunksize;
		const size_t extraItsPerPartition = (missalignment) ? chunksize - missalignment : 0;
		itsPerPartition += extraItsPerPartition;
		
		bounds_t partialBounds;
		partialBounds.step = step;
		partialBounds.chunksize = chunksize;
		
		// Start the partition from the original lower bound
		size_t lowerBound = originalLowerBound;
		
		for (int partition = 0; partition < numPartitions; ++partition) {
			// Compute the upper bound for this partition
			size_t upperBound = (partition < numPartitions - 1) ?
				lowerBound + itsPerPartition * step : originalUpperBound;
			
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
		return 1 + (((bounds.upper_bound - bounds.lower_bound) - 1) / bounds.step);
	}
};

#endif
