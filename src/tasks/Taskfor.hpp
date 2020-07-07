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
	static const int PENDING_CHUNKS_SIZE=7;
	static const int NUM_UINT64_BITS=64;
	// Global counter of remaining chunks
	std::atomic<int64_t> _remainingChunks;
	// Array of size_t to complete a cache line, where each bit of each size_t represents a chunk
	// _remainingChunks+_pendingChunks must occupy a single cache line in the most conservative
	// cache line size we found that is 64. So, we have _remainingChunks + 7 size_t in _pendingChunks.
	// That enables us to represent up to 448 chunks.
	std::atomic<uint64_t> _pendingChunks[PENDING_CHUNKS_SIZE];
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
		const TaskDataAccessesInfo &taskAccessInfo,
		const TaskHardwareCountersInfo &taskCounters,
		bool runnable = false
	) :
		Task(argsBlock, argsBlockSize, taskInfo,
			taskInvokationInfo, parent,
			instrumentationTaskId, flags,
			taskAccessInfo, taskCounters),
		  _remainingChunks(), _remainingIterations(),
		  _bounds(), _completedIterations(0),
		  _myChunk(-1)
	{
		assert(isFinal());
		setRunnable(runnable);

		std::atomic_init(&_remainingChunks, (int64_t) 0);
		std::atomic_init(_remainingIterations.ptr_to_basetype(), (size_t) 0);
		for (int i = 0; i < PENDING_CHUNKS_SIZE; i++) {
			std::atomic_init(&_pendingChunks[i], (uint64_t) 0);
		}
	}

	inline void setRunnable(bool runnableValue)
	{
		_flags[Task::non_runnable_flag] = !runnableValue;
	}

	inline size_t getIterationCount() const
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
			_bounds.chunksize = std::max(ceil(totalIterations, maxCollaborators), (size_t) 1);
		} else {
			// Distribute iterations over collaborators respecting the "alignment".
			size_t newChunksize = std::max(totalIterations / maxCollaborators, _bounds.chunksize);
			size_t alignedChunksize = closestMultiple(newChunksize, _bounds.chunksize);
			if (ceil(totalIterations, alignedChunksize) < maxCollaborators) {
				alignedChunksize = std::max(alignedChunksize - _bounds.chunksize, _bounds.chunksize);
			}
			assert(alignedChunksize % _bounds.chunksize == 0);
			_bounds.chunksize = alignedChunksize;
		}

		size_t totalChunks = ceil(totalIterations, _bounds.chunksize);
		// Each bit of the _pendingChunks var represents a chunk. 1 is pending, 0 is already executed.
		FatalErrorHandler::failIf(totalChunks > PENDING_CHUNKS_SIZE * NUM_UINT64_BITS, "Too many chunks required.");
		_remainingChunks.store(totalChunks, std::memory_order_relaxed);
		while (totalChunks > 0 ) {
			int index = (totalChunks-1)/NUM_UINT64_BITS;
			int localChunks = totalChunks - (index * NUM_UINT64_BITS);
			_pendingChunks[index].store((uint64_t)((uint64_t) 2 << (localChunks-1)) - (uint64_t) 1, std::memory_order_relaxed);
			totalChunks -= localChunks;
		}
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

	inline int getNextChunk(long cpuId, bool *remove = nullptr)
	{
		assert(!isRunnable());

		size_t totalIterations = _bounds.upper_bound - _bounds.lower_bound;
		size_t totalChunks = ceil(totalIterations, _bounds.chunksize);
		int chunkId = -1;
		size_t fetched = 0;

		int64_t remaining = _remainingChunks.fetch_sub(1, std::memory_order_relaxed) - 1;

		if (remaining < 0) {
			if (remove != nullptr)
				*remove = true;

			return chunkId;
		} else if (remove != nullptr) {
			*remove = (remaining == 0);
		}

		// Get the variable where we should start from based on cpuId.
		for (int c = 0; c < PENDING_CHUNKS_SIZE; c++) {
			do {
				int index = ((cpuId/NUM_UINT64_BITS) + c) % PENDING_CHUNKS_SIZE;
				std::atomic<uint64_t> &pendingChunks = _pendingChunks[index];

				// We use a right rotation to get the first enabled bit starting from cpuId.
				uint64_t aux = rotateRight(pendingChunks, (cpuId % totalChunks));
				// Find first set, -1 means no enabled bits at all.
				int ffs = indexFirstEnabledBit(aux);
				if (ffs == -1) {
					break;
				}

				// Adjust the id based on the rotation done
				chunkId = (ffs + (cpuId % totalChunks)) % NUM_UINT64_BITS;
				assert(chunkId < (int) NUM_UINT64_BITS);

				// Try to disable chunkId bit
				fetched = pendingChunks.fetch_and(~((uint64_t) 1 << chunkId), std::memory_order_relaxed);

				// If fetched is 0, there are no more bits enabled.
				if (fetched == 0) {
					break;
				}

				// If disable was successful, it means that chunk was not run and we must run it
				if (fetched & ((uint64_t) 1 << chunkId)) {
					chunkId += (NUM_UINT64_BITS * index);
					assert(chunkId < (int) totalChunks);
					return chunkId;
				}
			} while (1);
		}

		FatalErrorHandler::failIf(1, "There must be a chunk because remainingChunks assigned us a chunk.");
		return -1;
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

	inline void body(nanos6_address_translation_entry_t * = nullptr)
	{
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

	inline int getMyChunk() const
	{
		assert(isRunnable());
		return _myChunk;
	}

	inline size_t computeChunkBounds(size_t totalChunks, bounds_t const &sourceBounds)
	{
		assert(isRunnable());
		assert(totalChunks > 0);

		// Invert to start from the beginning.
		_myChunk = totalChunks - (_myChunk + 1);
		assert(_myChunk <= (int) totalChunks);
		assert(_myChunk >= 0);

		_bounds.lower_bound = sourceBounds.lower_bound + (_myChunk * sourceBounds.chunksize);
		size_t myIterations = std::min(sourceBounds.chunksize, sourceBounds.upper_bound - _bounds.lower_bound);
		assert(myIterations > 0 && myIterations <= sourceBounds.chunksize);
		_bounds.upper_bound = _bounds.lower_bound + myIterations;

		return myIterations;
	}

	inline size_t getCompletedIterations() const
	{
		assert(isRunnable());
		assert(getParent()->isTaskfor());
		return _completedIterations;
	}

	inline bool hasFirstChunk() const
	{
		assert(isRunnable());
		return (_bounds.lower_bound == 0);
	}

	inline bool hasLastChunk() const
	{
		assert(isRunnable());
		const Taskfor *source = (Taskfor *) getParent();
		return (_bounds.upper_bound == source->getBounds().upper_bound);
	}

	virtual inline void registerDependencies(bool = false)
	{
		assert(getParent() != nullptr);

		if (getParent()->isTaskloop()) {
			getTaskInfo()->register_depinfo(getArgsBlock(), (void *) &getBounds(), this);
		} else {
			getTaskInfo()->register_depinfo(getArgsBlock(), nullptr, this);
		}
	}

	virtual inline bool isDisposable() const
	{
		return !isRunnable();
	}

	virtual inline bool isTaskforCollaborator() const
	{
		return isRunnable();
	}

	virtual inline bool isTaskforSource() const
	{
		return !isRunnable();
	}

private:
	void run(Taskfor &source);

	static inline size_t closestMultiple(size_t n, size_t multipleOf)
	{
		return ((n + multipleOf - 1) / multipleOf) * multipleOf;
	}

	static inline size_t ceil(size_t x, size_t y)
	{
		return (x+(y-1))/y;
	}

	// Function to right rotate x by y bits
	static inline uint64_t rotateRight(uint64_t x, int y)
	{
		y &= 63;
		return (x >> y) | (x << (-y & 63));
	}

	// ffs returns the least signficant enabled bit, starting from 1
	// 0 means x has no enabled bits
	static inline int indexFirstEnabledBit(uint64_t x)
	{
		return __builtin_ffsll(x) - 1;
	}
};

#endif // TASKFOR_HPP
