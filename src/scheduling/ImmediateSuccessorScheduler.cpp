#include "ImmediateSuccessorScheduler.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "hardware/places/CPUPlace.hpp"
#include "tasks/Task.hpp"

#include <algorithm>
#include <cassert>
#include <mutex>

#define _unused(x) ((void)(x))

ImmediateSuccessorScheduler::ImmediateSuccessorScheduler() : SchedulerInterface()
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


ComputePlace * ImmediateSuccessorScheduler::addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint)
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


void ImmediateSuccessorScheduler::taskGetsUnblocked(Task *unblockedTask, ComputePlace *hardwarePlace)
{
    _unused(hardwarePlace);
	std::lock_guard<SpinLock> guard(_globalLock);
	_unblockedTasks.push_front(unblockedTask);
}


Task *ImmediateSuccessorScheduler::getReadyTask(ComputePlace *hardwarePlace, __attribute__((unused)) Task *currentTask)
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


ComputePlace *ImmediateSuccessorScheduler::getIdleComputePlace(bool force)
{
	std::lock_guard<SpinLock> guard(_globalLock);
	if (force || !_readyTasks.empty() || !_unblockedTasks.empty()) {
		return getIdleCPU();
	} else {
		return nullptr;
	}
}


void ImmediateSuccessorScheduler::disableComputePlace(ComputePlace *hardwarePlace)
{
	if (hardwarePlace->_schedulerData != nullptr) {
		Task *task = (Task *) hardwarePlace->_schedulerData;
		hardwarePlace->_schedulerData = nullptr;
		
		std::lock_guard<SpinLock> guard(_globalLock);
		_readyTasks.push_front(task);
	}
}

void ImmediateSuccessorScheduler::createReadyQueues(std::size_t nodes)
{
    _unused(nodes);
}
