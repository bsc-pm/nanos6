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
	ThreadManager::cpu_list_t const &cpuList = ThreadManager::getCPUListReference();
	for (auto it = cpuList.begin(); it != cpuList.end(); it++) {
		CPU *cpu = (*it).load();
		_immediateSuccessors[cpu] = nullptr;
	}
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


HardwarePlace * ImmediateSuccessorScheduler::addReadyTask(Task *task, __attribute__((unused)) HardwarePlace *hardwarePlace)
{
	CPU *cpu = (CPU *) hardwarePlace;
	auto position = _immediateSuccessors.find(cpu);
	assert(position != _immediateSuccessors.end());
	
	if (position->second == nullptr) {
		position->second = task;
		return nullptr;
	}
	
	std::lock_guard<SpinLock> guard(_globalLock);
	_readyTasks.push_front(task);
	
	return getIdleCPU();
}


void ImmediateSuccessorScheduler::taskGetsUnblocked(Task *unblockedTask, __attribute__((unused)) HardwarePlace *hardwarePlace)
{
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


Task *ImmediateSuccessorScheduler::getReadyTask(__attribute__((unused)) HardwarePlace *hardwarePlace, __attribute__((unused)) Task *currentTask)
{
	Task *task = nullptr;
	
	std::lock_guard<SpinLock> guard(_globalLock);
	
	// 1. Get an unblocked task
	task = getReplacementTask((CPU *) hardwarePlace);
	if (task != nullptr) {
		return task;
	}
	
	// 2. Get the immediate successor
	{
		CPU *cpu = (CPU *) hardwarePlace;
		auto position = _immediateSuccessors.find(cpu);
		assert(position != _immediateSuccessors.end());
		
		if (position->second != nullptr) {
			task = position->second;
			position->second = nullptr;
			return task;
		}
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

