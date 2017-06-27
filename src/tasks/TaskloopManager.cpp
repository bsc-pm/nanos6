#include <nanos6.h>
#include "Task.hpp"
#include "TaskloopInfo.hpp"
#include "TaskloopManager.hpp"
#include "TaskloopManagerImplementation.hpp"

void TaskloopManager::handleTaskloop(Taskloop *runnableTaskloop, Taskloop *sourceTaskloop)
{
	assert(runnableTaskloop != nullptr);
	assert(sourceTaskloop != nullptr);
	
	const nanos_task_info &taskInfo = *(runnableTaskloop->getTaskInfo());
	void *argsBlock = runnableTaskloop->getArgsBlock();
	
	size_t currentCPUId = 0;
	{
		WorkerThread *currentThread = runnableTaskloop->getThread();
		assert(currentThread != nullptr);
    	CPU *currentCPU = currentThread->getComputePlace();
		assert(currentCPU != nullptr);
		currentCPUId = currentCPU->_virtualCPUId;
	}
	
	const size_t partitionCount = sourceTaskloop->getPartitionCount();
	size_t partitionId = (currentCPUId) % partitionCount;
	
	bounds_t &bounds = runnableTaskloop->getTaskloopInfo().getBounds();
	
	while (true) {
		bool work = sourceTaskloop->getPendingIterationsFromPartition(partitionId, bounds);
		if (work) {
			taskInfo.run(argsBlock, &bounds);
		} else {
			if (!sourceTaskloop->hasPendingIterations()) {
				sourceTaskloop->notifyNoPendingIterations();
				return;
			}
			
			partitionId = (partitionId + 1) % partitionCount;
		}
	}
}

