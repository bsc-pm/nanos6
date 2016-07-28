#include "ImmediateSuccessorScheduler.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "hardware/places/CPUPlace.hpp"
#include "tasks/Task.hpp"

#include <algorithm>
#include <cassert>
#include <mutex>


ImmediateSuccessorScheduler::ImmediateSuccessorScheduler()
{
}

ImmediateSuccessorScheduler::~ImmediateSuccessorScheduler()
{
}


Task *ImmediateSuccessorScheduler::getReplacementTask(__attribute__((unused)) CPU *hardwarePlace)
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


void ImmediateSuccessorScheduler::cpuBecomesIdle(CPU *cpu)
{
	_idleCPUs.push_front(cpu);
}


CPU *ImmediateSuccessorScheduler::getIdleCPU()
{
	if (!_idleCPUs.empty()) {
		CPU *idleCPU = _idleCPUs.front();
		_idleCPUs.pop_front();
		
		return idleCPU;
	}
	
	return nullptr;
}


HardwarePlace * ImmediateSuccessorScheduler::addReadyTask(Task *task, HardwarePlace *hardwarePlace, ReadyTaskHint hint)
{
	// The following condition is only needed for the "main" task, that is added by something that is not a hardware place and thus should end up in a queue
	if (hardwarePlace != nullptr) {
		if ((hint != CHILD_TASK_HINT) && (hardwarePlace->_schedulerData == nullptr)) {
			hardwarePlace->_schedulerData = task;
			return nullptr;
		}
	}
	
	std::lock_guard<SpinLock> guard(_globalLock);
	_readyTasks.push_front(task);
	
	return getIdleCPU();
}


void ImmediateSuccessorScheduler::taskGetsUnblocked(Task *unblockedTask, HardwarePlace *hardwarePlace)
{
	assert(hardwarePlace != nullptr);
	if (hardwarePlace->_schedulerData == nullptr) {
		hardwarePlace->_schedulerData = unblockedTask;
		return;
	}
	
	std::lock_guard<SpinLock> guard(_globalLock);
	_unblockedTasks.push_front(unblockedTask);
}


bool ImmediateSuccessorScheduler::checkIfIdleAndGrantReactivation(HardwarePlace *hardwarePlace)
{
	std::lock_guard<SpinLock> guard(_globalLock);
	
	auto it = std::find(_idleCPUs.begin(), _idleCPUs.end(), (CPU *) hardwarePlace);
	
	if (it != _idleCPUs.end()) {
		_idleCPUs.erase(it);
		return true;
	}
	
	return false;
}


Task *ImmediateSuccessorScheduler::getReadyTask(HardwarePlace *hardwarePlace, __attribute__((unused)) Task *currentTask)
{
	Task *task = nullptr;
	
	// 1. Get the immediate successor
	if (hardwarePlace->_schedulerData != nullptr) {
		task = (Task *) hardwarePlace->_schedulerData;
		hardwarePlace->_schedulerData = nullptr;
		return task;
	}
	
	std::lock_guard<SpinLock> guard(_globalLock);
	
	// 2. Get an unblocked task
	task = getReplacementTask((CPU *) hardwarePlace);
	if (task != nullptr) {
		return task;
	}
	
	// 3. Or get a ready task
	if (!_readyTasks.empty()) {
		task = _readyTasks.front();
		_readyTasks.pop_front();
		
		assert(task != nullptr);
		
		return task;
	}
	
	// 4. Or mark the CPU as idle
	cpuBecomesIdle((CPU *) hardwarePlace);
	
	return nullptr;
}


HardwarePlace *ImmediateSuccessorScheduler::getIdleHardwarePlace(bool force)
{
	std::lock_guard<SpinLock> guard(_globalLock);
	if (force || !_readyTasks.empty() || !_unblockedTasks.empty()) {
		return getIdleCPU();
	} else {
		return nullptr;
	}
}

