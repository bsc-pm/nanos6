#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/HardwarePlace.hpp"
#include "scheduling/Scheduler.hpp"
#include <tasks/Task.hpp>

#include <cassert>


namespace ompss {

void addTask(Task *task)
{
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != nullptr);
	
	HardwarePlace *hardwarePlace = currentWorkerThread->getHardwarePlace();
	assert(currentWorkerThread->getTask() != nullptr);
	task->setParent(currentWorkerThread->getTask());
	
	Scheduler::addChildTask(task, hardwarePlace);
	ThreadManager::resumeAnyIdle(hardwarePlace);
}


} // namespace ompss
