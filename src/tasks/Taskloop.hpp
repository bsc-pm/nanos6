#ifndef TASKLOOP_HPP
#define TASKLOOP_HPP

#include <iostream>

#include "tasks/Task.hpp"
#include "tasks/TaskloopInfo.hpp"

class Taskloop : public Task {
private:
	size_t _argsBlockSize;
	
	TaskloopInfo _taskloopInfo;
	
public:
	typedef nanos6_taskloop_bounds_t bounds_t;
	
	inline Taskloop(
		void *argsBlock, size_t argsBlockSize,
		nanos_task_info *taskInfo,
		nanos_task_invocation_info *taskInvokationInfo,
		Task *parent,
		Instrument::task_id_t instrumentationTaskId,
		size_t flags,
		bool runnable = false
	)
		: Task(argsBlock, taskInfo, taskInvokationInfo, parent, instrumentationTaskId, flags),
		_argsBlockSize(argsBlockSize),
		_taskloopInfo()
	{
		setRunnable(runnable);
		setArgsBlockOwner(true);
	}
	
	inline void setArgsBlockSize(size_t argsBlockSize)
	{
		_argsBlockSize = argsBlockSize;
	}
	
	inline size_t getArgsBlockSize() const
	{
		return _argsBlockSize;
	}
	
	inline TaskloopInfo const &getTaskloopInfo() const
	{
		return _taskloopInfo;
	}
	
	inline TaskloopInfo &getTaskloopInfo()
	{
		return _taskloopInfo;
	}
	
	inline void body()
	{
		assert(hasCode());
		assert(isRunnable());
		assert(_thread != nullptr);
		
		Task *parent = getParent();
		assert(parent != nullptr);
		assert(parent->isTaskloop());
		
		run(*((Taskloop *)parent));
	}
	
	inline bool markAsFinished() __attribute__((warn_unused_result))
	{
		bool runnable = isRunnable();
		if (runnable) {
			assert(_thread != nullptr);
			_thread = nullptr;
		}
		
		int countdown = decreaseAndGetRemovalBlockingCount();
		assert(countdown >= 0);
		
		if (!runnable) {
			TaskDataAccesses &accessStructures = getDataAccesses();
			assert(!accessStructures.hasBeenDeleted());
			
			if (!accessStructures._accesses.empty() && countdown == 1) {
				unregisterDataAccesses();
			}
		}
		
		return (countdown == 0);
	}
	
	//! \brief Remove a nested task (because it has finished)
	//!
	//! \returns true iff the change makes this task become ready or disposable
	inline bool removeChild(__attribute__((unused)) Task *child) __attribute__((warn_unused_result))
	{
		int countdown = decreaseAndGetRemovalBlockingCount();
		assert(countdown >= 0);
		
		if (!isRunnable()) {
			TaskDataAccesses &accessStructures = getDataAccesses();
			assert(!accessStructures.hasBeenDeleted());
			
			if (!accessStructures._accesses.empty() && countdown == 1) {
				unregisterDataAccesses();
			}
		}
		
		return (countdown == 0);
	}
	
	inline void setRunnable(bool runnableValue)
	{
		_flags[Task::non_runnable_flag] = !runnableValue;
	}
	
	inline bool isRunnable() const
	{
		return !_flags[Task::non_runnable_flag];
	}
	
	void setArgsBlockOwner(bool argsBlockOwnerValue)
	{
		_flags[Task::non_owner_args_flag] = !argsBlockOwnerValue;
	}
	
	bool isArgsBlockOwner() const
	{
		return !_flags[Task::non_owner_args_flag];
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
	
	inline size_t getPartitionCount()
	{
		return _taskloopInfo.getPartitionCount();
	}
	
	inline bool getPendingIterationsFromPartition(int partitionId, bounds_t &obtainedBounds)
	{
		assert(!isRunnable());
		assert(partitionId >= 0);
		assert(partitionId < CPUS_PER_PARTITION);
		
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
	
private:
	void run(Taskloop &source);
	
	void unregisterDataAccesses();
	
};

#endif // TASKLOOP_HPP
