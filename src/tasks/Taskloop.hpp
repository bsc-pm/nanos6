/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKLOOP_HPP
#define TASKLOOP_HPP

#include <iostream>

#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"
#include "tasks/TaskloopInfo.hpp"

#define MOD(a, b)  ((a) < 0 ? ((((a) % (b)) + (b)) % (b)) : ((a) % (b)))

class Taskloop : public Task {
private:
	TaskloopInfo _taskloopInfo;
	
public:
	typedef nanos6_taskloop_bounds_t bounds_t;
	
	inline Taskloop(
		void *argsBlock, size_t argsBlockSize,
		nanos6_task_info_t *taskInfo,
		nanos6_task_invocation_info_t *taskInvokationInfo,
		Task *parent,
		Instrument::task_id_t instrumentationTaskId,
		size_t flags,
		bool runnable = false
	)
		: Task(argsBlock, argsBlockSize, taskInfo, taskInvokationInfo, parent, instrumentationTaskId, flags),
		_taskloopInfo()
	{
		setRunnable(runnable);
		setDelayedRelease(true);
	}
	
	inline TaskloopInfo const &getTaskloopInfo() const
	{
		return _taskloopInfo;
	}
	
	inline TaskloopInfo &getTaskloopInfo()
	{
		return _taskloopInfo;
	}
	
	inline void body(
		__attribute__((unused)) void *deviceEnvironment,
		__attribute__((unused)) nanos6_address_translation_entry_t *translationTable = nullptr
	) {
		assert(hasCode());
		assert(isRunnable());
		assert(_thread != nullptr);
		assert(deviceEnvironment == nullptr);
		
		Task *parent = getParent();
		assert(parent != nullptr);
		assert(parent->isTaskloop());
		
		run(*((Taskloop *)parent));
	}
	
	inline void setRunnable(bool runnableValue)
	{
		_flags[Task::non_runnable_flag] = !runnableValue;
	}
	
	inline void setDelayedRelease(bool delayedReleaseValue)
	{
		_flags[Task::wait_flag] = delayedReleaseValue;
	}
	
	inline bool hasPendingIterations()
	{
		assert(!isRunnable());
		
		return (_taskloopInfo._remainingPartitions.load() > 0);
	}
	
	inline void notifyCollaboratorHasStarted()
	{
		assert(!isRunnable());
		
		increaseRemovalBlockingCount();
	}
	
	inline void notifyCollaboratorHasFinished()
	{
		assert(!isRunnable());
		
		decreaseRemovalBlockingCount();
	}

private:
	inline int getPartitionCount()
	{
		return _taskloopInfo.getPartitionCount();
	}
	
	inline bool isDistributionFunctionEnabled()
	{
		EnvironmentVariable<int> distributionFunction("NANOS6_TASKLOOP_DISTRIBUTION_FUNCTION", 0);
		int value = distributionFunction.getValue();
		assert(value >= 0);
		
		return (value > 0);
	}
	
	inline bool getPendingIterationsFromPartition(int partitionId, bounds_t &obtainedBounds)
	{
		assert(!isRunnable());
		assert(partitionId >= 0);
		assert(partitionId < getPartitionCount());
		
		bounds_t &bounds = _taskloopInfo._bounds;
		const size_t step = bounds.step;
		const size_t chunksize = bounds.chunksize;
		const size_t steppedChunksize = step * chunksize;
		
		TaskloopPartition &partition = _taskloopInfo._partitions[partitionId];
		const size_t originalUpperBound = partition.upperBound;
		
		const size_t lowerBound = std::atomic_fetch_add(&(partition.nextLowerBound), steppedChunksize);
		
		if (lowerBound < originalUpperBound) {
			const size_t upperBound = std::min(lowerBound + steppedChunksize, originalUpperBound);
			
			obtainedBounds.lower_bound = lowerBound;
			obtainedBounds.upper_bound = upperBound;
			obtainedBounds.chunksize = chunksize;
			obtainedBounds.step = step;
			
			if (upperBound >= originalUpperBound) {
				--_taskloopInfo._remainingPartitions;
			}
			
			return true;
		}
		
		return false;
	}
	
	void getPartitionPath(int CPUId, std::vector<int> &partitionPath);
	
	void run(Taskloop &source);
};

#endif // TASKLOOP_HPP
