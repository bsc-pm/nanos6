#include <nanos6.h>
#include "Task.hpp"
#include "TaskloopBounds.hpp"
#include "TaskloopInfo.hpp"
#include "TaskloopManager.hpp"
#include "TaskloopManagerImplementation.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/ComputePlace.hpp"

#include <DataAccessRegistration.hpp>

void TaskloopManager::handleTaskloop(Taskloop *runnableTaskloop, Taskloop *sourceTaskloop)
{
	assert(runnableTaskloop != nullptr);
	assert(sourceTaskloop != nullptr);
	
	TaskloopInfo &taskloopInfo = runnableTaskloop->getTaskloopInfo();
	nanos6_taskloop_bounds_t &bounds = taskloopInfo._bounds;
	nanos_task_info *taskInfo = runnableTaskloop->getTaskInfo();
	void *argsBlock = runnableTaskloop->getArgsBlock();
	
	bool assignedWork;
	do {
		taskInfo->run(argsBlock, &bounds);
		
		assignedWork = sourceTaskloop->getIterations(false, bounds);
	} while (assignedWork);
}

void TaskloopManager::unregisterTaskloopDataAccesses(Taskloop *taskloop)
{
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != nullptr);
	
	ComputePlace *computePlace = currentWorkerThread->getComputePlace();
	assert(computePlace != nullptr);
	
	DataAccessRegistration::unregisterTaskDataAccesses(taskloop, computePlace);
}

