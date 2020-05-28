/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
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
	size_t myIterations = computeChunkBounds();
	assert(myIterations > 0);
	size_t completedIterations = 0;

	do {
		taskInfo.implementations[0].run(argsBlock, &bounds, nullptr);

		completedIterations += myIterations;

		__attribute__ ((unused)) bool placeHolder;
		_myChunk = source.getNextChunk(getThread()->getComputePlace()->getIndex(), placeHolder);
		if (_myChunk >= 0) {
			myIterations = computeChunkBounds();
		} else {
			myIterations = 0;
		}
	} while (myIterations != 0);

	assert(completedIterations > 0);
	assert(completedIterations <= source._bounds.upper_bound);
	_completedIterations = completedIterations;

	source.notifyCollaboratorHasFinished();
}
