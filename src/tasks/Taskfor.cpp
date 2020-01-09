/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#include "Taskfor.hpp"
#include "executors/threads/WorkerThread.hpp"

void Taskfor::run(Taskfor &source)
{
	assert(getParent()->isTaskfor() && getParent() == &source);
	assert(getMyChunk() >= 0);

	// Temporary hack in order to solve the problem of updating
	// the location of the DataAccess objects of the Taskfor,
	// when we unregister them, until we solve this properly,
	// by supporting the Taskfor construct through the execution
	// workflow
	MemoryPlace *memoryPlace = getThread()->getComputePlace()->getMemoryPlace(0);
	source.setMemoryPlace(memoryPlace);

	// Get the arguments and the task information
	const nanos6_task_info_t &taskInfo = *getTaskInfo();
	void *argsBlock = getArgsBlock();
	bounds_t &bounds = getBounds();
	computeChunkBounds(bounds);
	size_t myIterations = getIterationCount();
	assert(myIterations > 0);
	size_t completedIterations = 0;

	do {
		taskInfo.implementations[0].run(argsBlock, &bounds, nullptr);
		bounds.lower_bound = bounds.upper_bound;

		completedIterations += myIterations;

		_myChunk = source.getNextChunk();
		if (_myChunk >= 0) {
			computeChunkBounds(bounds);
			myIterations = getIterationCount();
		} else {
			myIterations = 0;
		}
	} while (myIterations != 0);

	assert(completedIterations > 0);
	_completedIterations = completedIterations;

	source.notifyCollaboratorHasFinished();
}
