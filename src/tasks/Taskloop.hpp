/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
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
	bool _sourceTaskloop;

public:
	inline Taskloop(
		void *argsBlock, size_t argsBlockSize,
		nanos6_task_info_t *taskInfo,
		nanos6_task_invocation_info_t *taskInvokationInfo,
		Task *parent,
		Instrument::task_id_t instrumentationTaskId,
		size_t flags
	)
		: Task(argsBlock, argsBlockSize, taskInfo, taskInvokationInfo, parent, instrumentationTaskId, flags, nullptr, nullptr, 0),
		  _bounds(), _sourceTaskloop(false)
	{}

	inline void initialize(size_t lowerBound, size_t upperBound, size_t grainsize)
	{
		_bounds.lower_bound = lowerBound;
		_bounds.upper_bound = upperBound;
		_bounds.grainsize = grainsize;
		_sourceTaskloop = true;

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

	inline bool isSourceTaskloop() const
	{
		return _sourceTaskloop;
	}

	inline bool hasPendingIterations()
	{
		return (getIterationCount() > 0);
	}

	void body(
	__attribute__((unused)) void *deviceEnvironment,
	__attribute__((unused)) nanos6_address_translation_entry_t *translationTable = nullptr);
};

#endif // TASKLOOP_HPP
