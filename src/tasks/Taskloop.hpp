#ifndef TASKLOOP_HPP
#define TASKLOOP_HPP

#include <iostream>

#include "tasks/Task.hpp"
#include "tasks/TaskloopBounds.hpp"
#include "tasks/TaskloopInfo.hpp"
#include "tasks/TaskloopManager.hpp"

class Taskloop : public Task {
private:
	size_t _argsBlockSize;
	
	TaskloopInfo _taskloopInfo;
	
public:
	Taskloop(
		void *argsBlock, size_t argsBlockSize,
		nanos_task_info *taskInfo,
		nanos_task_invocation_info *taskInvokationInfo,
		nanos_taskloop_bounds *taskloopBounds,
		Task *parent,
		Instrument::task_id_t instrumentationTaskId,
		size_t flags,
		bool runnable = false
	)
		: Task(argsBlock, taskInfo, taskInvokationInfo, parent, instrumentationTaskId, flags),
		_argsBlockSize(argsBlockSize),
		_taskloopInfo(taskloopBounds)
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
		
		Task *parent = getParent();
		assert(parent != nullptr);
		assert(parent->isTaskloop());
		
		TaskloopManager::handleTaskloop(this, (Taskloop *)parent);
	}
	
	inline bool markAsFinished() __attribute__((warn_unused_result))
	{
		if (isRunnable()) {
			assert(_thread != nullptr);
			_thread = nullptr;
		}
		
		return decreaseRemovalBlockingCount();
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
	
	inline bool getIterations(size_t completedIterations, nanos_taskloop_bounds *partialBounds, bool *complete = nullptr)
	{
		assert(partialBounds != nullptr);
		assert(!isRunnable());
		
		nanos_taskloop_bounds *bounds = _taskloopInfo._bounds;
		assert(bounds != nullptr);
		
		const size_t originalLowerBound = bounds->lower_bound;
		const size_t originalUpperBound = bounds->upper_bound;
		const size_t gridSize = bounds->grid_size;
		const size_t step = bounds->step;
		
		const size_t remainingIterations = (_taskloopInfo._remainingIterations -= completedIterations);
		
		if (remainingIterations != 0) {
			size_t chunkSize = gridSize * step;
			size_t lowerBound = std::atomic_fetch_add(&_taskloopInfo._nextLowerBound, chunkSize);
			
			if (lowerBound < originalUpperBound) {
				size_t upperBound = std::min(lowerBound + chunkSize, originalUpperBound);
				
				partialBounds->lower_bound = lowerBound;
				partialBounds->upper_bound = upperBound;
				partialBounds->grid_size = gridSize;
				partialBounds->step = step;
				
				if (complete != nullptr) {
					*complete = (upperBound == originalUpperBound);
				}
			
				// The first to get a chunk
				if (lowerBound == originalLowerBound) {
					increaseRemovalBlockingCount();
				}
				
				return true;
			}
		} else if (completedIterations > 0) {
			decreaseRemovalBlockingCount();
		}
		
		if (complete != nullptr) {
			*complete = true;
		}
		
		return false;
	}
};

#endif // TASKLOOP_HPP
