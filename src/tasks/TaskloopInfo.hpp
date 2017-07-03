#ifndef TASKLOOP_INFO_HPP
#define TASKLOOP_INFO_HPP

#include <atomic>
#include <cstdlib>

#include <nanos6.h>
#include "TaskloopLogic.hpp"
#include "executors/threads/CPUManager.hpp"
#include "lowlevel/EnvironmentVariable.hpp"

#define CACHE_LINE_SIZE 128
#define CPUS_PER_PARTITION 4

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
		
		size_t partitionCount = getPartitionCount();
		assert(partitionCount > 0);
		
		// Allocate memory for the partitions
		posix_memalign((void **)&_partitions, CACHE_LINE_SIZE, sizeof(TaskloopPartition) * partitionCount);
		assert(_partitions != nullptr);
		
		// Get the partitions
		std::vector<bounds_t> partialBounds;
		TaskloopLogic::splitIterations(partitionCount, _bounds, partialBounds);
		assert(partialBounds.size() == partitionCount);
		
		// Set the partitions
		for (size_t i = 0; i < partitionCount; ++i) {
			_partitions[i].upperBound = partialBounds[i].upper_bound;
			_partitions[i].nextLowerBound = partialBounds[i].lower_bound;
		}
		
		// Initialize the number of remaining partitions
		_remainingPartitions = partitionCount;
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
	
	inline size_t getPartitionCount()
	{
		static size_t partitionCount = 1 + ((CPUManager::getTotalCPUs() - 1) / getCPUsPerPartition());
		return partitionCount;
	}
	
	static inline size_t getCPUsPerPartition()
	{
		EnvironmentVariable<size_t> cpusPerPartition("NANOS6_CPUS_PER_TASKLOOP_PARTITION", CPUS_PER_PARTITION);
		
		return cpusPerPartition.getValue();
	}
};

#endif // TASKLOOP_INFO_HPP
