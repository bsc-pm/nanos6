#include <nanos6.h>
#include "Task.hpp"
#include "TaskloopBounds.hpp"
#include "TaskloopInfo.hpp"
#include "TaskloopManager.hpp"
#include "TaskloopManagerImplementation.hpp"


void TaskloopManager::handleTaskloop(Taskloop *runnableTaskloop, Taskloop *sourceTaskloop)
{
	assert(runnableTaskloop != nullptr);
	assert(sourceTaskloop != nullptr);
	
	TaskloopInfo &taskloopInfo = runnableTaskloop->getTaskloopInfo();
	nanos_taskloop_bounds *bounds = taskloopInfo._bounds;
	nanos_task_info *taskInfo = runnableTaskloop->getTaskInfo();
	void *argsBlock = runnableTaskloop->getArgsBlock();
	
	bool assignedWork;
	do {
		taskInfo->run(argsBlock, bounds);
		
		size_t completedIterations = TaskloopBounds::getIterationCount(bounds);
		assert(completedIterations > 0);
		
		assignedWork = sourceTaskloop->getIterations(completedIterations, bounds);
	} while (assignedWork);
}

