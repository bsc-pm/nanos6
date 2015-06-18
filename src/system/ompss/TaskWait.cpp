#include "TaskWait.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"

#include <cassert>


namespace ompss {

void taskWait()
{
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != nullptr);
	
	Task *task = currentWorkerThread->getTask();
	assert(task != nullptr);
	
	if (!task->doesNotNeedToBlockForChildren()) {
		if (!task->markAsBlocked() ) {
			ThreadManager::suspendForBlocking(currentWorkerThread);
			assert(task->canBeWokenUp());
		}
		task->markAsUnblocked();
	}
}


} // namespace ompss
