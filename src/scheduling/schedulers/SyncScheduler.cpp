/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/


#include "SyncScheduler.hpp"
#include "scheduling/schedulers/device/DeviceScheduler.hpp"
#include "scheduling/schedulers/device/SubDeviceScheduler.hpp"

Task *SyncScheduler::getTask(ComputePlace *computePlace, ComputePlace *deviceComputePlace, bool device, bool subdevice)
{
	assert(!(device && subdevice));
	
	if (computePlace == nullptr) {
		_lock.lock();
		Task *myTask = nullptr;
		if (deviceComputePlace != nullptr) {
			myTask = _scheduler->getReadyTask(deviceComputePlace);
		}
		else {
			myTask = _scheduler->getReadyTask(computePlace);
		}
		_lock.unsubscribe();
		assert(myTask == nullptr || myTask->isRunnable());
		return myTask;
	}
	
	assert(computePlace != nullptr);
	assert(computePlace->getType() == nanos6_host_device);
	
	uint64_t const cpuIndex = computePlace->getIndex();
	if (device) {
		((DeviceScheduler *)this)->setCPUToDevice(cpuIndex, deviceComputePlace);
	}
	if (subdevice) {
		((SubDeviceScheduler *)this)->setCPUToSubDevice(cpuIndex, deviceComputePlace);
	}
	
	// Subscribe to the lock.
	uint64_t const myTicket = _lock.subscribeOrLock(cpuIndex);
	Task *task;
	
	if (getAssignedTask(cpuIndex, myTicket, task)) {
		// Someone got the lock and gave me work to do.
		assert(task->isRunnable());
		return task;
	}
	
	// I own the lock!
	// First of all, get all the tasks in the addQueues into the ready queue.
	processReadyTasks();
	
	uint64_t cpu;
	uint64_t i = myTicket+1;
	const std::vector<CPU *> &computePlaces = CPUManager::getCPUListReference();
	
	// Serve all the subscribers, while there is work to give them.
	while (_lock.popWaitingCPU(i, cpu)) {
		ComputePlace *resultComputePlace = nullptr;
		if (device) {
			resultComputePlace = ((DeviceScheduler *)this)->getCPUToDevice(cpuIndex);
		}
		else if (subdevice) {
			resultComputePlace = ((SubDeviceScheduler *)this)->getCPUToSubDevice(cpuIndex);
		}
		else {
			resultComputePlace = computePlaces[cpu];
		}
		assert(resultComputePlace != nullptr);
		
		Task *const localTask = _scheduler->getReadyTask(resultComputePlace);
		if (localTask == nullptr)
			break;
		assert(localTask->isRunnable());
		
		if (device) {
			((DeviceScheduler *)this)->setCPUToDevice(cpuIndex, nullptr);
		}
		if (subdevice) {
			((SubDeviceScheduler *)this)->setCPUToSubDevice(cpuIndex, nullptr);
		}
		
		// Put a task into the subscriber slot.
		assignTask(cpu, i, localTask);
		
		// Advance the ticket of the subscriber just served.
		_lock.unsubscribe();
		i++;
	};
	
	// No more subscribers. Try to get work for myself.
	Task *myTask = nullptr;
	if (deviceComputePlace != nullptr) {
		myTask = _scheduler->getReadyTask(deviceComputePlace);
	}
	else {
		myTask = _scheduler->getReadyTask(computePlace);
	}
	_lock.unsubscribe();
	assert(myTask == nullptr || myTask->isRunnable());
	return myTask;
}
