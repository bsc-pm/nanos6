/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
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
	// Source
	Padded<std::atomic<int>> _currentChunk;
	// Source
	Padded<std::atomic<size_t>> _remainingIterations;
	// Source and collaborator
	bounds_t _bounds;
	// Collaborator
	size_t _completedIterations;
	// Collaborator
	int _myChunk;

public:
	// Methods for both source and collaborator taskfors
	inline Taskfor(
		void *argsBlock, size_t argsBlockSize,
		nanos6_task_info_t *taskInfo,
		nanos6_task_invocation_info_t *taskInvokationInfo,
		Task *parent,
		Instrument::task_id_t instrumentationTaskId,
		size_t flags,
		TaskDataAccessesInfo taskAccessInfo,
		const TaskHardwareCounters &taskCounters,
		bool runnable = false
	)
		: Task(argsBlock, argsBlockSize, taskInfo, taskInvokationInfo, parent, instrumentationTaskId, flags, taskAccessInfo, taskCounters),
		  _currentChunk(), _remainingIterations(), _bounds(), _completedIterations(0), _myChunk(-1)
	{
		assert(isFinal());
		setRunnable(runnable);

		std::atomic_init(_currentChunk.ptr_to_basetype(), (int) 0);
		std::atomic_init(_remainingIterations.ptr_to_basetype(), (size_t) 0);
	}

	inline void setRunnable(bool runnableValue)
	{
		_flags[Task::non_runnable_flag] = !runnableValue;
	}

	inline size_t getIterationCount()
	{
		return (_bounds.upper_bound - _bounds.lower_bound);
	}

	// Methods for source taskfors
	inline void initialize(size_t lowerBound, size_t upperBound, size_t chunksize)
	{
		assert(!isRunnable());

		_bounds.lower_bound = lowerBound;
		_bounds.upper_bound = upperBound;
		_bounds.chunksize = chunksize;

		size_t maxCollaborators = CPUManager::getNumCPUsPerTaskforGroup();
		assert(maxCollaborators > 0);

		size_t totalIterations = getIterationCount();
		_remainingIterations.store(totalIterations, std::memory_order_relaxed);

		if (_bounds.chunksize == 0) {
			// Just distribute iterations over collaborators if no hint.
			_bounds.chunksize = std::max(totalIterations / maxCollaborators, (size_t) 1);
		}
		else {
			// Distribute iterations over collaborators respecting the "alignment".
			size_t newChunksize = std::max(totalIterations / maxCollaborators, _bounds.chunksize);
			size_t alignedChunksize = closestMultiple(newChunksize, _bounds.chunksize);
			if (std::ceil((double)totalIterations / (double)alignedChunksize) < maxCollaborators) {
				alignedChunksize = std::max(alignedChunksize - _bounds.chunksize, _bounds.chunksize);
			}
			assert(alignedChunksize % _bounds.chunksize == 0);
			_bounds.chunksize = alignedChunksize;
		}

		assert(_currentChunk == 0);
		_currentChunk.store(std::ceil((double) totalIterations/(double) _bounds.chunksize), std::memory_order_relaxed);
	}

	inline bounds_t const &getBounds() const
	{
		assert(!isRunnable());
		return _bounds;
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

	inline void markAsScheduled()
	{
		assert(!isRunnable());

		increaseRemovalBlockingCount();
	}

	inline bool removedFromScheduler()
	{
		assert(!isRunnable());

		return decreaseRemovalBlockingCount();
	}

	inline bool decrementRemainingIterations(size_t amount)
	{
		assert(!isRunnable());
		size_t remaining = _remainingIterations.fetch_sub(amount, std::memory_order_relaxed) - amount;
		return (remaining == 0);
	}

	inline int getNextChunk()
	{
		assert(!isRunnable());
		int myChunk = _currentChunk.fetch_sub(1, std::memory_order_relaxed) - 1;
		return myChunk;
	}

	// Methods for collaborator taskfors
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
		assert(isRunnable());
		Task::reinitialize(argsBlock, argsBlockSize, taskInfo, taskInvokationInfo, parent, instrumentationTaskId, flags);
		_bounds.lower_bound = 0;
		_bounds.upper_bound = 0;
		_bounds.grainsize = 0;
		_bounds.chunksize = 0;
		_completedIterations = 0;
		setRunnable(runnable);
	}

	inline void body(
		__attribute__((unused)) nanos6_address_translation_entry_t *translationTable = nullptr
	) {
		assert(hasCode());
		assert(isRunnable());
		assert(_thread != nullptr);

		Task *parent = getParent();
		assert(parent != nullptr);
		assert(parent->isTaskfor());

		run(*((Taskfor *)parent));
	}

	inline bounds_t &getBounds()
	{
		return _bounds;
	}

	inline void setChunk(int chunk)
	{
		assert(isRunnable());
		_myChunk = chunk;
	}

	inline int getMyChunk()
	{
		assert(isRunnable());
		return _myChunk;
	}

	inline size_t computeChunkBounds()
	{
		assert(isRunnable());
		bounds_t &collaboratorBounds = _bounds;
		const Taskfor *source = (Taskfor *) getParent();
		bounds_t const &sourceBounds = source->getBounds();
		size_t totalIterations = sourceBounds.upper_bound - sourceBounds.lower_bound;
		size_t totalChunks = std::ceil((double) totalIterations / (double) sourceBounds.chunksize);
		assert(totalChunks > 0);
		_myChunk = totalChunks - (_myChunk + 1);

		collaboratorBounds.lower_bound = sourceBounds.lower_bound + (_myChunk * sourceBounds.chunksize);
		size_t myIterations = std::min(sourceBounds.chunksize, sourceBounds.upper_bound - collaboratorBounds.lower_bound);
		assert(myIterations > 0 && myIterations <= sourceBounds.chunksize);
		collaboratorBounds.upper_bound = collaboratorBounds.lower_bound + myIterations;

		return myIterations;
	}

	inline size_t getCompletedIterations()
	{
		assert(isRunnable());
		assert(getParent()->isTaskfor());
		return _completedIterations;
	}

	inline bool hasFirstChunk()
	{
		assert(isRunnable());
		return (_bounds.lower_bound == 0);
	}

	inline bool hasLastChunk()
	{
		assert(isRunnable());
		const Taskfor *source = (Taskfor *) getParent();
		return (_bounds.upper_bound == source->getBounds().upper_bound);
	}

	virtual inline void registerDeps(bool = false)
	{
		assert(getParent() != nullptr);

		if (getParent()->isTaskloop()) {
			getTaskInfo()->register_depinfo(getArgsBlock(), (void *) &getBounds(), this);
		} else {
			getTaskInfo()->register_depinfo(getArgsBlock(), nullptr, this);
		}
	}

	virtual inline bool isDisposable()
	{
		return !isRunnable();
	}

	virtual inline bool isTaskforCollaborator()
	{
		return isRunnable();
	}

	virtual inline bool isSourceTaskfor()
	{
		return !isRunnable();
	}

private:
	void run(Taskfor &source);

	static inline size_t closestMultiple(size_t n, size_t multipleOf)
	{
		return ((n + multipleOf - 1) / multipleOf) * multipleOf;
	}
};

#endif // TASKFOR_HPP
