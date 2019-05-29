/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_IMPLEMENTATION_HPP
#define TASK_IMPLEMENTATION_HPP

#include "Task.hpp"

#include <DataAccessRegistration.hpp>
#include <InstrumentTaskId.hpp>
#include <TaskDataAccessesImplementation.hpp>


inline Task::Task(
	void *argsBlock,
	size_t argsBlockSize,
	nanos6_task_info_t *taskInfo,
	nanos6_task_invocation_info_t *taskInvokationInfo,
	Task *parent,
	Instrument::task_id_t instrumentationTaskId,
	size_t flags
)
	: _argsBlock(argsBlock),
	_argsBlockSize(argsBlockSize),
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
	_memoryPlace(nullptr),
	_countdownToRelease(1),
	_workflow(nullptr),
	_executionStep(nullptr),
	_taskStatistics(),
	_taskPredictions(),
	_taskCounters(),
	_taskCountersPredictions(),
	_clusterContext(nullptr)
{
	if (parent != nullptr) {
		parent->addChild(this);
	}
}

inline Task::~Task()
{
	if (_clusterContext != nullptr) {
		delete _clusterContext;
	}
}

inline bool Task::markAsFinished(ComputePlace *computePlace)
{
	CPUDependencyData hpDependencyData;
	
	// Non-runnable taskloops should avoid these checks
	if (isRunnable()) {
		if (_taskInfo->implementations[0].device_type_id == nanos6_device_t::nanos6_host_device) {
			// Not true anymore, a Task might have been offloaded
			// to a remote device, in which case it wouldn't have
			// a thread assigned to it.
			// assert(_thread != nullptr);
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
		//! We need to pass 'nullptr' here as a ComputePlace to notify
		//! the DataAccessRegistration system that it is creating
		//! taskwait fragments for a 'wait' task.
		DataAccessRegistration::handleEnterTaskwait(this, nullptr, hpDependencyData);
		
		if (!decreaseRemovalBlockingCount()) {
			return false;
		}
		
		// All its children are completed, so the delayed release of
		// dependencies has successfully completed
		completeDelayedRelease();
		DataAccessRegistration::handleExitTaskwait(this, computePlace, hpDependencyData);
		increaseRemovalBlockingCount();
	}
	
	// Return whether all external events have been also fulfilled, so
	// the dependencies can be released
	return decreaseReleaseCount();
}

// Return if the task can release its dependencies
inline bool Task::markAllChildrenAsFinished(ComputePlace *computePlace)
{
	assert(_thread == nullptr);
	
	CPUDependencyData hpDependencyData;
	
	// Complete the delayed release of dependencies
	completeDelayedRelease();
	DataAccessRegistration::handleExitTaskwait(this, computePlace, hpDependencyData);
	increaseRemovalBlockingCount();
	
	// Return whether all external events have been also fulfilled, so
	// the dependencies can be released
	return decreaseReleaseCount();
}

#endif // TASK_IMPLEMENTATION_HPP
