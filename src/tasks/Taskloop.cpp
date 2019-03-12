/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "Taskloop.hpp"
#include "TaskloopInfo.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "lowlevel/EnvironmentVariable.hpp"

#include <DataAccessRegistration.hpp>

template <class T = int>
static inline T modulus(T a, T b)
{
	return (a < 0) ? (((a % b) + b) % b) : (a % b);
}

void Taskloop::getPartitionPath(int CPUId, std::vector<int> &partitionPath)
{
	static bool useDistributionFunction = isDistributionFunctionEnabled();
	
	int partitionCount = getPartitionCount();
	int partition = CPUId % partitionCount;
	
	if (useDistributionFunction) {
		int sign = 1;
		for (int i = 0; i < partitionCount; ++i) {
			partition = modulus(partition + sign * i, partitionCount);
			partitionPath.push_back(partition);
			sign *= -1;
		}
	} else {
		for (int i = 0; i < partitionCount; ++i) {
			partitionPath.push_back(partition);
			partition = (partition + 1) % partitionCount;
		}
	}
	assert((int) partitionPath.size() == partitionCount);
}

void Taskloop::run(Taskloop &source)
{
	// Get the arguments and the task information
	const nanos6_task_info_t &taskInfo = *getTaskInfo();
	void *argsBlock = getArgsBlock();
	bounds_t &bounds = _taskloopInfo.getBounds();
	
	// Get the number of partitions
	const int partitionCount = source.getPartitionCount();
	
	// Get the current CPU
	CPU *currentCPU = getThread()->getComputePlace();
	assert(currentCPU != nullptr);
	
	/* Temporary hack in order to solve the problem of updating
	 * the location of the DataAccess objects of the Taskloop,
	 * when we unregister them, until we solve this properly,
	 * by supporting the Taskloop construct through the execution
	 * Workflow */
	MemoryPlace *memoryPlace = currentCPU->getMemoryPlace(0);
	source.setMemoryPlace(memoryPlace);
	
	// Get the path of partitions to vist
	std::vector<int> partitionPath;
	getPartitionPath(currentCPU->_virtualCPUId, partitionPath);
	
	// Get the initial partition identifier
	int partitionId = partitionPath[0];
	int visitedPartitions = 0;
	
	while (true) {
		// Try to get a chunk of iterations
		bool work = source.getPendingIterationsFromPartition(partitionId, bounds);
		if (work) {
			taskInfo.implementations[0].run(argsBlock, &bounds, nullptr);
		} else {
			++visitedPartitions;
			
			// Finalize the execution in case there are no iterations
			if (!source.hasPendingIterations() || visitedPartitions == partitionCount) {
				source.notifyCollaboratorHasFinished();
				return;
			}
			assert(visitedPartitions < partitionCount);
			
			// Move to the next partition
			partitionId = partitionPath[visitedPartitions];
		}
	}
}

