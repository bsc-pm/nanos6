/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKFOR_HPP
#define TASKFOR_HPP

#include <cmath>

#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"
#include "tasks/TaskforInfo.hpp"

#define MOD(a, b)  ((a) < 0 ? ((((a) % (b)) + (b)) % (b)) : ((a) % (b)))

class Taskfor : public Task {
private:
	TaskforInfo _taskforInfo;
	
public:
	typedef nanos6_taskfor_bounds_t bounds_t;
	
	inline Taskfor(
		void *argsBlock, size_t argsBlockSize,
		nanos6_task_info_t *taskInfo,
		nanos6_task_invocation_info_t *taskInvokationInfo,
		Task *parent,
		Instrument::task_id_t instrumentationTaskId,
		size_t flags,
		bool precreated = false,
		bool runnable = false
	)
		: Task(argsBlock, argsBlockSize, taskInfo, taskInvokationInfo, parent, instrumentationTaskId, flags, nullptr, nullptr, 0),
		_taskforInfo(precreated)
	{
		assert(!runnable);
		assert(isFinal());
		setRunnable(runnable);
		setDelayedRelease(true);
	}
	
	inline void reinitialize(
		void *argsBlock, size_t argsBlockSize,
		nanos6_task_info_t *taskInfo,
		nanos6_task_invocation_info_t *taskInvokationInfo,
		Task *parent,
		Instrument::task_id_t instrumentationTaskId,
		size_t flags,
		bool runnable = false
	)
	{
		Task::reinitialize(argsBlock, argsBlockSize, taskInfo, taskInvokationInfo, parent, instrumentationTaskId, flags);
		_taskforInfo.reinitialize();
		setRunnable(runnable);
	}
	
	inline TaskforInfo const &getTaskforInfo() const
	{
		return _taskforInfo;
	}
	
	inline TaskforInfo &getTaskforInfo()
	{
		return _taskforInfo;
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
		assert(parent->isTaskfor());
		assert(((Taskfor *)parent)->_taskforInfo._remainingIterations.load() > 0);
		
		run(*((Taskfor *)parent));
	}
	
	inline void setRunnable(bool runnableValue)
	{
		_flags[Task::non_runnable_flag] = !runnableValue;
	}
	
	inline bool hasPendingIterations()
	{
		assert(!isRunnable());
		
		return (_taskforInfo._bounds.upper_bound > _taskforInfo._bounds.lower_bound);
	}
	
	inline void notifyCollaboratorHasStarted()
	{
		assert(!isRunnable());
		
		increaseRemovalBlockingCount();
	}
	
	inline bool notifyCollaboratorHasFinished()
	{
		assert(!isRunnable());
		
		return decreaseRemovalBlockingCount();
	}
	
	inline bool getChunks(bounds_t &collaboratorBounds)
	{
		assert(!isRunnable());
		
		bounds_t &bounds = _taskforInfo._bounds;
		size_t totalIterations = bounds.upper_bound - bounds.lower_bound;
		assert(totalIterations > 0);
		size_t totalChunks = std::ceil((double) totalIterations / (double) bounds.chunksize);
		assert(totalChunks > 0);
		assert(_taskforInfo._maxCollaborators > 0);
		size_t maxCollaborators = _taskforInfo._maxCollaborators--;
		assert(maxCollaborators > 0);
		size_t myChunks = std::ceil((double) totalChunks/(double) maxCollaborators);
		assert(myChunks > 0);
		size_t myIterations = std::min(myChunks*bounds.chunksize, bounds.upper_bound - bounds.lower_bound);
		assert(myIterations > 0);
		
		collaboratorBounds.lower_bound = bounds.lower_bound;
		collaboratorBounds.upper_bound = collaboratorBounds.lower_bound + myIterations;
		
		bounds.lower_bound += myIterations;
		
		bool lastChunks = (collaboratorBounds.upper_bound == bounds.upper_bound);
		assert(lastChunks || bounds.lower_bound < bounds.upper_bound);
		
		return lastChunks;
	}
	
	inline bool decrementRemainingIterations(size_t amount)
	{
		return _taskforInfo.decrementRemainingIterations(amount);
	}
	
	inline size_t getCompletedIterations()
	{
		assert(getParent()->isTaskfor());
		return _taskforInfo.getCompletedIterations();
	}
	
	inline bool hasFirstChunk()
	{
		return (_taskforInfo.getBounds().lower_bound == 0);
	}
	
	inline bool hasLastChunk()
	{
		return (_taskforInfo.getBounds().upper_bound == ((Taskfor *) getParent())->getTaskforInfo().getBounds().upper_bound);
	}
	
private:
	void run(Taskfor &source);
};

#endif // TASKFOR_HPP
