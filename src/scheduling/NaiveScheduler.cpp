#include "NaiveScheduler.hpp"
#include "hardware/places/CPUPlace.hpp"

#include <cassert>
#include <mutex>


NaiveScheduler::NaiveScheduler()
{
}

NaiveScheduler::~NaiveScheduler()
{
}


void NaiveScheduler::addMainTask(Task *mainTask)
{
	// No locking needed since the thread manager has not been started yet
	_readyTasks.push_front(mainTask);
}


void NaiveScheduler::addSiblingTask(Task *newReadyTask, __attribute__((unused)) Task *triggererTask, __attribute__((unused)) HardwarePlace const *hardwarePlace)
{
	std::lock_guard<SpinLock> guard(_readyTasksLock);
	_readyTasks.push_front(newReadyTask);
}


void NaiveScheduler::addChildTask(Task *newReadyTask, __attribute__((unused)) HardwarePlace const *hardwarePlace)
{
	std::lock_guard<SpinLock> guard(_readyTasksLock);
	_readyTasks.push_front(newReadyTask);
}

Task *NaiveScheduler::schedule(__attribute__((unused)) HardwarePlace const *hardwarePlace)
{
	Task *task = nullptr;
	
	std::lock_guard<SpinLock> guard(_readyTasksLock);
	if (!_readyTasks.empty()) {
		task = _readyTasks.front();
		_readyTasks.pop_front();
	}
	
	return task;
}

