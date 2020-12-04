/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKLOOP_HPP
#define TASKLOOP_HPP

#include <cmath>

#include "support/MathSupport.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"


class Taskloop : public Task {
public:
	typedef nanos6_loop_bounds_t bounds_t;

private:
	bounds_t _bounds;

	bool _source;

	// In some cases, the compiler cannot precisely indicate the number of deps.
	// In these cases, it passes -1 to the runtime so the deps are dynamically
	// registered. We have a loop where the parent registers all the deps of the
	// child tasks. We can count on that loop how many deps has each child and
	// get the max, so when we create children, we can use a more refined
	// numDeps, saving memory space and probably improving slightly the performance.
	size_t _maxChildDeps;

public:
	inline Taskloop(
		void *argsBlock,
		size_t argsBlockSize,
		nanos6_task_info_t *taskInfo,
		nanos6_task_invocation_info_t *taskInvokationInfo,
		Task *parent,
		Instrument::task_id_t instrumentationTaskId,
		size_t flags,
		const TaskDataAccessesInfo &taskAccessInfo,
		void *taskCountersAddress,
		void *taskStatistics
	) :
		Task(argsBlock, argsBlockSize,
			taskInfo, taskInvokationInfo,
			parent, instrumentationTaskId,
			flags, taskAccessInfo,
			taskCountersAddress,
			taskStatistics),
		_bounds(),
		_source(false),
		_maxChildDeps(0)
	{
	}

	inline void initialize(size_t lowerBound, size_t upperBound, size_t grainsize, size_t chunksize)
	{
		_bounds.lower_bound = lowerBound;
		_bounds.upper_bound = upperBound;
		_bounds.grainsize = grainsize;
		_bounds.chunksize = chunksize;
		_source = true;

		size_t totalIterations = getIterationCount();

		// Set a implementation defined chunksize if needed
		if (_bounds.grainsize == 0) {
			_bounds.grainsize = std::max(totalIterations /CPUManager::getTotalCPUs(), (size_t) 1);
		}
	}

	inline bounds_t &getBounds()
	{
		return _bounds;
	}

	inline bounds_t const &getBounds() const
	{
		return _bounds;
	}

	inline size_t getIterationCount() const
	{
		return (_bounds.upper_bound - _bounds.lower_bound);
	}

	void body(nanos6_address_translation_entry_t *translationTable) override;

	inline void registerDependencies(bool discrete = false) override
	{
		if (discrete && isTaskloopSource()) {
			bounds_t tmpBounds;
			size_t numTasks = computeNumTasks(getIterationCount(), _bounds.grainsize);
			for (size_t t = 0; t < numTasks; t++) {
				// Store previous maxChildDeps
				size_t maxChildDepsStart = _maxChildDeps;
				// Reset
				_maxChildDeps = 0;

				// Register deps of children task
				tmpBounds.lower_bound = _bounds.lower_bound + t * _bounds.grainsize;
				tmpBounds.upper_bound = std::min(tmpBounds.lower_bound + _bounds.grainsize, _bounds.upper_bound);
				getTaskInfo()->register_depinfo(getArgsBlock(), (void *) &tmpBounds, this);

				// Restore previous maxChildDeps if it is bigger than current one
				if (maxChildDepsStart > _maxChildDeps) {
					_maxChildDeps = maxChildDepsStart;
				}
			}
			assert(tmpBounds.upper_bound == _bounds.upper_bound);
		} else {
			getTaskInfo()->register_depinfo(getArgsBlock(), (void *) &_bounds, this);
		}
	}

	inline size_t getMaxChildDependencies() const
	{
		return _maxChildDeps;
	}

	inline void increaseMaxChildDependencies() override
	{
		if (_source) {
			_maxChildDeps++;
		}
	}

	inline bool isTaskloopSource() const override
	{
		return _source;
	}

	inline bool isTaskloopFor() const override
	{
		return isTaskfor();
	}

	static inline size_t computeNumTasks(size_t iterations, size_t grainsize)
	{
		if (grainsize == 0) {
			grainsize = std::max(iterations / CPUManager::getTotalCPUs(), (size_t) 1);
		}
		return MathSupport::ceil(iterations, grainsize);
	}
};

#endif // TASKLOOP_HPP
