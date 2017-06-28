#include "Taskloop.hpp"
#include "TaskloopInfo.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/ComputePlace.hpp"

#include <DataAccessRegistration.hpp>

void Taskloop::run(Taskloop &source)
{
	// Get the arguments and the task information
	const nanos_task_info &taskInfo = *getTaskInfo();
	void *argsBlock = getArgsBlock();
	bounds_t &bounds = _taskloopInfo.getBounds();
	
	// Get the number of partitions
	const int partitionCount = source.getPartitionCount();
	
	// Get the current CPU
    CPU *currentCPU = getThread()->getComputePlace();
	assert(currentCPU != nullptr);
	
	// Compute the inital partition
	int partitionId = currentCPU->_virtualCPUId % partitionCount;
	
	while (true) {
		// Try to get a chunk of iterations
		bool work = source.getPendingIterationsFromPartition(partitionId, bounds);
		if (work) {
			taskInfo.run(argsBlock, &bounds);
		} else {
			// Finalize the execution in case there are no iterations
			if (!source.hasPendingIterations()) {
				source.notifyCollaboratorHasFinished();
				return;
			}
			
			// Reconsider the partition
			partitionId = (partitionId + 1) % partitionCount;
		}
	}
}

void Taskloop::unregisterDataAccesses()
{
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != nullptr);
	
	ComputePlace *computePlace = currentWorkerThread->getComputePlace();
	assert(computePlace != nullptr);
	
	DataAccessRegistration::unregisterTaskDataAccesses(this, computePlace);
}

