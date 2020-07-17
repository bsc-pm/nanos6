/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKLOOP_HPP
#define TASKLOOP_HPP

#include <cmath>

#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"


class Taskloop : public Task {
public:
	typedef nanos6_loop_bounds_t bounds_t;

private:
	bounds_t _bounds;
	bool _source;

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
		void *taskCountersAddress
	) :
		Task(argsBlock, argsBlockSize,
			taskInfo, taskInvokationInfo,
			parent, instrumentationTaskId,
			flags, taskAccessInfo,
			taskCountersAddress),
		_bounds(),
		_source(false)
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

	void body(nanos6_address_translation_entry_t * = nullptr) override;

	inline void registerDependencies(bool discrete = false) override
	{
		if (discrete && isTaskloopSource()) {
			size_t tasks = std::ceil((double) (_bounds.upper_bound - _bounds.lower_bound) / (double) _bounds.grainsize);
			bounds_t tmpBounds;
			for (size_t t = 0; t < tasks; t++) {
				tmpBounds.lower_bound = _bounds.lower_bound + t * _bounds.grainsize;
				tmpBounds.upper_bound = std::min(tmpBounds.lower_bound + _bounds.grainsize, _bounds.upper_bound);
				getTaskInfo()->register_depinfo(getArgsBlock(), (void *) &tmpBounds, this);
			}
			assert(tmpBounds.upper_bound == _bounds.upper_bound);
		} else {
			getTaskInfo()->register_depinfo(getArgsBlock(), (void *) &_bounds, this);
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
};

#endif // TASKLOOP_HPP
