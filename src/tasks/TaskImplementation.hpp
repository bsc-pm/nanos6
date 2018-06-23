/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_IMPLEMENTATION_HPP
#define TASK_IMPLEMENTATION_HPP


#include "Task.hpp"

#include <TaskDataAccessesImplementation.hpp>
#include <DataAccessRegistration.hpp>

#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>


inline Task::Task(
	void *argsBlock,
	nanos_task_info *taskInfo,
	nanos_task_invocation_info *taskInvokationInfo,
	Task *parent,
	Instrument::task_id_t instrumentationTaskId,
	size_t flags
)
	: _argsBlock(argsBlock),
	_taskInfo(taskInfo),
	_taskInvokationInfo(taskInvokationInfo),
	_countdownToBeWokenUp(1),
	_parent(parent),
	_priority(0),
	_thread(nullptr),
	_dataAccesses(),
	_flags(flags),
	_predecessorCount(0),
	_instrumentationTaskId(instrumentationTaskId),
	_schedulerInfo(nullptr),
	_computePlace(nullptr),
	_countdownToRelease(1)
{
	if (parent != nullptr) {
		parent->addChild(this);
	}
}

inline bool Task::markAsFinished(ComputePlace *computePlace)
{
	assert(computePlace != nullptr);
	
	// Non-runnable taskloops should avoid these checks
	if (isRunnable()) {
		if (_taskInfo->implementations[0].device_type_id == nanos6_device_t::nanos6_host_device) {
			assert(_thread != nullptr);
			_thread = nullptr;
		} else {
			assert(_computePlace != nullptr);
			_computePlace = nullptr;
		}
	}
	
	// If the task has a wait clause, the release of dependencies must be
	// delayed (at least) until the task finishes its execution and all
	// its children complete and become disposable
	if (mustDelayRelease()) {
		DataAccessRegistration::handleEnterTaskwait(this, computePlace);
		
		if (!decreaseRemovalBlockingCount()) {
			return false;
		}
		
		// All its children are completed, so the delayed release of
		// dependencies has successfully completed
		completeDelayedRelease();
		DataAccessRegistration::handleExitTaskwait(this, computePlace);
		increaseRemovalBlockingCount();
	}
	
	// Return whether all external events have been also fulfilled, so
	// the dependencies can be released
	return decreaseReleaseCount();
}

// Return if the task can release its dependencies
inline bool Task::markAllChildrenAsFinished(ComputePlace *computePlace)
{
	assert(computePlace != nullptr);
	assert(_thread == nullptr);
	assert(_computePlace == nullptr);
	
	// Complete the delayed release of dependencies
	completeDelayedRelease();
	DataAccessRegistration::handleExitTaskwait(this, computePlace);
	increaseRemovalBlockingCount();
	
	// Return whether all external events have been also fulfilled, so
	// the dependencies can be released
	return decreaseReleaseCount();
}

#endif // TASK_IMPLEMENTATION_HPP
