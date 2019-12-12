/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#include "Taskfor.hpp"
#include "executors/threads/WorkerThread.hpp"

void Taskfor::run(Taskfor &source)
{
	assert(getParent()->isTaskfor() && getParent() == &source);
	
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
	size_t myIterations = getIterationCount();
	
	size_t originalUpperBound = bounds.upper_bound;
	do {
		bounds.upper_bound = std::max(bounds.lower_bound + bounds.chunksize, originalUpperBound);
		taskInfo.implementations[0].run(argsBlock, &bounds, nullptr);
		bounds.lower_bound = bounds.upper_bound;
	} while (bounds.upper_bound < originalUpperBound);
	
	assert(bounds.upper_bound == originalUpperBound);
	
	_completedIterations = myIterations;
	source.notifyCollaboratorHasFinished();
}
