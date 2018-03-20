/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKLOOP_INFO_HPP
#define TASKLOOP_INFO_HPP

#include <atomic>
#include <cstdlib>

#include <nanos6.h>
#include "TaskloopLogic.hpp"
#include "executors/threads/CPUManager.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

#define CACHE_LINE_SIZE 128
#define DEFAULT_CPUS_PER_PARTITION 1

#ifndef ALIGNED
#define ALIGNED __attribute__ ((aligned (CACHE_LINE_SIZE)))
#endif

struct alignas(CACHE_LINE_SIZE) TaskloopPartition {
	size_t upperBound;
	std::atomic<size_t> nextLowerBound;
	
	TaskloopPartition()
		: upperBound(0),
		nextLowerBound(0)
	{
	}
};

struct TaskloopInfo {
private:
	typedef nanos6_taskloop_bounds_t bounds_t;
	
	friend class Taskloop;
	
protected:
	bounds_t _bounds;
	
	std::atomic<size_t> _remainingPartitions;
	
	TaskloopPartition *_partitions;
	
public:
	inline TaskloopInfo()
		: _bounds(),
		_remainingPartitions(0),
		_partitions(nullptr)
	{
	}
	
	inline ~TaskloopInfo()
	{
		if (_partitions != nullptr) {
			free(_partitions);
		}
	}
	
	inline void initialize()
	{
		assert(_partitions == nullptr);
		
		int partitionCount = getPartitionCount();
		assert(partitionCount > 0);
		
		// Set a implementation defined chunksize if needed
		if (_bounds.chunksize == 0) {
			size_t totalIterations = TaskloopLogic::getIterationCount(_bounds);
			_bounds.chunksize = std::max(totalIterations / (2 * partitionCount), (size_t) 1);
		}
		
		// Allocate memory for the partitions
		int rc = posix_memalign((void **)&_partitions, CACHE_LINE_SIZE, sizeof(TaskloopPartition) * partitionCount);
		FatalErrorHandler::handle(rc, " when trying to allocate memory for the new taskloop's partitions");
		assert(_partitions != nullptr);
		
		// Get the partitions
		std::vector<bounds_t> partialBounds;
		TaskloopLogic::splitIterations(partitionCount, _bounds, partialBounds);
		assert((int) partialBounds.size() == partitionCount);
		
		// Count the number of non-empty partitions
		int nonEmptyPartitions = 0;
		
		// Set the partitions
		for (int i = 0; i < partitionCount; ++i) {
			_partitions[i].upperBound = partialBounds[i].upper_bound;
			_partitions[i].nextLowerBound = partialBounds[i].lower_bound;
			
			if (TaskloopLogic::getIterationCount(partialBounds[i]) > 0) {
				++nonEmptyPartitions;
			}
		}
		
		// Initialize the number of remaining partitions
		_remainingPartitions = nonEmptyPartitions;
	}
	
	inline void initialize(size_t lowerBound, size_t upperBound, size_t step, size_t chunksize)
	{
		_bounds.lower_bound = lowerBound;
		_bounds.upper_bound = upperBound;
		_bounds.step = step;
		_bounds.chunksize = chunksize;
		
		initialize();
	}
	
	inline void initialize(const bounds_t &newBounds)
	{
		// Set the bounds
		_bounds.lower_bound = newBounds.lower_bound;
		_bounds.upper_bound = newBounds.upper_bound;
		_bounds.chunksize = newBounds.chunksize;
		_bounds.step = newBounds.step;
		
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
	
	inline int getPartitionCount()
	{
		static int partitionCount = 1 + ((CPUManager::getTotalCPUs() - 1) / getCPUsPerPartition());
		return partitionCount;
	}
	
	static inline int getCPUsPerPartition()
	{
		EnvironmentVariable<int> cpusPerPartition("NANOS6_CPUS_PER_TASKLOOP_PARTITION", DEFAULT_CPUS_PER_PARTITION);
		int value = cpusPerPartition.getValue();
		assert(value > 0);
		
		return value;
	}
};

#endif // TASKLOOP_INFO_HPP
