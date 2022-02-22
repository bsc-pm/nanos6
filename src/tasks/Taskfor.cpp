/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2022 Barcelona Supercomputing Center (BSC)
*/

#include "Taskfor.hpp"
#include "executors/threads/WorkerThread.hpp"

void Taskfor::run(Taskfor &source, nanos6_address_translation_entry_t *translationTable)
{
	assert(getParent()->isTaskfor() && getParent() == &source);
	assert(getMyChunk() >= 0);

	// Temporary hack in order to solve the problem of updating
	// the location of the DataAccess objects of the Taskfor,
	// when we unregister them, until we solve this properly,
	// by supporting the Taskfor construct through the execution
	// workflow
	ComputePlace *computePlace = getThread()->getComputePlace();
	int cpuId = computePlace->getIndex();
	MemoryPlace *memoryPlace = computePlace->getMemoryPlace(0);
	source.setMemoryPlace(memoryPlace);

	// Compute source taskfor total chunks
	bounds_t const &sourceBounds = source.getBounds();
	const size_t totalIterations = sourceBounds.upper_bound - sourceBounds.lower_bound;
	const size_t totalChunks = MathSupport::ceil(totalIterations, sourceBounds.chunksize);

	// Get the arguments and the task information
	const nanos6_task_info_t &taskInfo = *getTaskInfo();
	void *argsBlock = getArgsBlock();
	size_t myIterations = computeChunkBounds(totalChunks, sourceBounds);
	assert(myIterations > 0);
	size_t completedIterations = 0;

	do {
		taskInfo.implementations[0].run(argsBlock, &_bounds, translationTable);

		completedIterations += myIterations;

		_myChunk = source.getNextChunk(cpuId);
		if (_myChunk >= 0) {
			myIterations = computeChunkBounds(totalChunks, sourceBounds);
		} else {
			myIterations = 0;
		}
	} while (myIterations != 0);

	assert(completedIterations > 0);
	assert(completedIterations <= source._bounds.upper_bound);
	_completedIterations = completedIterations;

	source.notifyCollaboratorHasFinished();
}
