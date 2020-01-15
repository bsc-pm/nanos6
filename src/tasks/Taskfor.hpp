/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKFOR_HPP
#define TASKFOR_HPP

#include <cmath>

#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

class Taskfor : public Task {
public:
	typedef nanos6_loop_bounds_t bounds_t;

private:
	bounds_t _bounds;
	size_t _maxCollaborators;
	std::atomic<size_t> _remainingIterations;
	size_t _completedIterations;

public:
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
		  _bounds(), _remainingIterations(0), _completedIterations(0)
	{
		assert(!runnable);
		assert(isFinal());
		setRunnable(runnable);
		setDelayedRelease(true);
		_maxCollaborators = precreated ? 0 : CPUManager::getNumCPUsPerTaskforGroup();
		assert(precreated || CPUManager::getNumCPUsPerTaskforGroup() > 0);
		assert(precreated || _maxCollaborators > 0);
	}

	inline void initialize(size_t lowerBound, size_t upperBound, size_t chunksize)
	{
		_bounds.lower_bound = lowerBound;
		_bounds.upper_bound = upperBound;
		_bounds.chunksize = chunksize;

		assert(_maxCollaborators > 0);
		size_t totalIterations = getIterationCount();
		_remainingIterations = totalIterations;

		// Set a implementation defined chunksize if needed
		if (_bounds.chunksize == 0) {
			_bounds.chunksize = std::max(totalIterations / (_maxCollaborators), (size_t) 1);
		}
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
		_bounds = bounds_t();
		_remainingIterations = 0;
		_completedIterations = 0;
		setRunnable(runnable);
	}

	inline bounds_t &getBounds()
	{
		return _bounds;
	}

	inline bounds_t const &getBounds() const
	{
		return _bounds;
	}

	inline void setBounds(bounds_t &bounds)
	{
		_bounds.lower_bound = bounds.lower_bound;
		_bounds.upper_bound = bounds.upper_bound;
	}

	inline size_t getIterationCount() const
	{
		return (_bounds.upper_bound - _bounds.lower_bound);
	}

	inline bool decrementRemainingIterations(size_t amount)
	{
		size_t remaining = (_remainingIterations -= amount);
		return (remaining == 0);
	}

	inline size_t getCompletedIterations() const
	{
		return _completedIterations;
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
		assert(((Taskfor *)parent)->_remainingIterations.load() > 0);

		run(*((Taskfor *)parent));
	}

	inline void setRunnable(bool runnableValue)
	{
		_flags[Task::non_runnable_flag] = !runnableValue;
	}

	inline bool hasPendingIterations()
	{
		assert(!isRunnable());

		return (_bounds.upper_bound > _bounds.lower_bound);
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

		bounds_t &bounds = _bounds;
		size_t totalIterations = bounds.upper_bound - bounds.lower_bound;
		assert(totalIterations > 0);
		size_t totalChunks = std::ceil((double) totalIterations / (double) bounds.chunksize);
		assert(totalChunks > 0);
		assert(_maxCollaborators > 0);
		size_t maxCollaborators = _maxCollaborators--;
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

	inline bool hasFirstChunk()
	{
		return (_bounds.lower_bound == 0);
	}

	inline bool hasLastChunk()
	{
		return (_bounds.upper_bound == ((Taskfor *) getParent())->getBounds().upper_bound);
	}

private:
	void run(Taskfor &source);
};

#endif // TASKFOR_HPP
