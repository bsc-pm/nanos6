#include "NaiveScheduler.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "hardware/places/CPUPlace.hpp"
#include "tasks/Task.hpp"

#include <algorithm>
#include <cassert>
#include <mutex>


NaiveScheduler::NaiveScheduler()
{
}

NaiveScheduler::~NaiveScheduler()
{
}


Task *NaiveScheduler::getReplacementTask(__attribute__((unused)) CPU *hardwarePlace)
{
	if (!_unblockedTasks.empty()) {
		Task *replacementTask = _unblockedTasks.front();
		_unblockedTasks.pop_front();
		
		assert(replacementTask != nullptr);
		
		return replacementTask;
	} else {
		return nullptr;
	}
}


void NaiveScheduler::cpuBecomesIdle(CPU *cpu)
{
	_idleCPUs.push_front(cpu);
}


CPU *NaiveScheduler::getIdleCPU()
{
	if (!_idleCPUs.empty()) {
		CPU *idleCPU = _idleCPUs.front();
		_idleCPUs.pop_front();
		
		return idleCPU;
	}
	
	return nullptr;
}


HardwarePlace * NaiveScheduler::addReadyTask(Task *task, __attribute__((unused)) HardwarePlace *hardwarePlace, __attribute__((unused)) ReadyTaskHint hint)
{
	std::lock_guard<SpinLock> guard(_globalLock);
	_readyTasks.push_front(task);
	
	return getIdleCPU();
}


void NaiveScheduler::taskGetsUnblocked(Task *unblockedTask, __attribute__((unused)) HardwarePlace *hardwarePlace)
{
	std::lock_guard<SpinLock> guard(_globalLock);
	_unblockedTasks.push_front(unblockedTask);
}


Task *NaiveScheduler::getReadyTask(__attribute__((unused)) HardwarePlace *hardwarePlace, __attribute__((unused)) Task *currentTask)
{
	Task *task = nullptr;
	
	std::lock_guard<SpinLock> guard(_globalLock);
	
	// 1. Get an unblocked task
	task = getReplacementTask((CPU *) hardwarePlace);
	if (task != nullptr) {
		return task;
	}
	
	// 2. Or get a ready task
	if (!_readyTasks.empty()) {
		task = _readyTasks.front();
		_readyTasks.pop_front();
		
		assert(task != nullptr);
		
		return task;
	}
	
	// 3. Or mark the CPU as idle
	cpuBecomesIdle((CPU *) hardwarePlace);
	
	return nullptr;
}


HardwarePlace *NaiveScheduler::getIdleHardwarePlace(bool force)
{
	std::lock_guard<SpinLock> guard(_globalLock);
	if (force || !_readyTasks.empty() || !_unblockedTasks.empty()) {
		return getIdleCPU();
	} else {
		return nullptr;
	}
}

