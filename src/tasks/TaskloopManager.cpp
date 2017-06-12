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
	nanos6_taskloop_bounds_t &bounds = taskloopInfo._bounds;
	nanos_task_info *taskInfo = runnableTaskloop->getTaskInfo();
	void *argsBlock = runnableTaskloop->getArgsBlock();
	
	bool assignedWork;
	do {
		taskInfo->run(argsBlock, &bounds);
		
		assignedWork = sourceTaskloop->getIterations(false, bounds);
	} while (assignedWork);
}

