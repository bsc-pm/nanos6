/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#include <string>

#include <nanos6/polling.h>

#include "DeviceComputePlace.hpp"
#include "executors/threads/TaskFinalization.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/HardwareInfo.hpp"
#include "hardware/device/DeviceInfoImplementation.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/TaskImplementation.hpp"

#include <DataAccessRegistration.hpp>
#include <ExecutionWorkflow.hpp>

DeviceComputePlace::DeviceComputePlace(DeviceMemoryPlace *memoryPlace,
		nanos6_device_t type, int subType, int index, void *deviceHandler) :
		ComputePlace(index, type), _type(type), _subType(subType), _deviceHandler(
				deviceHandler), _maxRunningTasks(deviceMaxRunningTask(type)), _runningTasks(0), _strPollingService(
				std::string("runner:") + std::to_string(type) + " " + std::to_string(index) + " "
						+ std::to_string(subType)), _functions(
				HardwareInfo::getDeviceFunctions(type)), _memoryPlace(memoryPlace)
{
	activatePollingService();
}

DeviceComputePlace::~DeviceComputePlace()
{
	deactivatePollingService();
}

int DeviceComputePlace::pollingFinishTasks(DeviceFunctionsInterface *functions)
{
	std::vector<Task *> taskvec;
	functions->getFinishedTasks(taskvec);
	
	for (Task *task : taskvec) {
		CPUDependencyData localHpDependencyData;
		if (task->markAsFinished(task->getComputePlace())) {
			DataAccessRegistration::unregisterTaskDataAccesses(task,
			task->getComputePlace(), localHpDependencyData, task->getMemoryPlace(), true);
			if (task->markAsReleased()) {
				TaskFinalization::disposeOrUnblockTask(task, task->getComputePlace());
			}
		}
	}
	
	return false;
}

int DeviceComputePlace::pollingRun(DeviceComputePlace *computePlace)
{
	int i = 0;
	while (computePlace->canRunTask()) {
		if (i++ > computePlace->getMaxRunningTasks())
			return false;
		
		WorkerThread *currentThread =  WorkerThread::getCurrentWorkerThread();
		if (currentThread == nullptr)
			break;
		auto cpu = currentThread->getComputePlace();
		
		Task *task = Scheduler::getReadyTask(cpu, computePlace);
		
		if (task == nullptr)
			continue;
		
		task->setComputePlace(computePlace);
		computePlace->runTask(task);
		
	}
	return false;
}

int DeviceComputePlace::getMaxRunningTasks()
{
	return _maxRunningTasks;
}
void DeviceComputePlace::runTask(Task *task)
{
	_runningTasks++;
	task->setComputePlace(this);
	task->setMemoryPlace((MemoryPlace *) _memoryPlace);
	_functions->setDevice(_index);
	void *extraData = _functions->generateDeviceExtra(task, _deviceHandler);
	// nanos6_address_translation_entry_t translates[0];
	// TODO: Do not pass an empty translation table since Mercurium
	// uses it to translate data addresses if it is not null
	task->body(extraData);
	_functions->postBodyDevice(task, extraData);
}

void DeviceComputePlace::disposeTask()
{
	_runningTasks--;
}

bool DeviceComputePlace::canRunTask()
{
	return (getRunningTasks() < _maxRunningTasks);
}

int DeviceComputePlace::getRunningTasks()
{
	return _runningTasks;
}

DeviceMemoryPlace *DeviceComputePlace::getMemoryPlace()
{
	return _memoryPlace;
}

int DeviceComputePlace::getSubType()
{
	return _subType;
}

int DeviceComputePlace::getType()
{
	return _type;
}

void DeviceComputePlace::deactivatePollingService()
{
	nanos6_unregister_polling_service(_strPollingService.c_str(),
			(nanos6_polling_service_t) pollingRun, this);
}

void DeviceComputePlace::activatePollingService()
{
	nanos6_register_polling_service(_strPollingService.c_str(),
			(nanos6_polling_service_t) pollingRun, this);
}

int DeviceComputePlace::deviceMaxRunningTask(nanos6_device_t dev)
{
	static int nanos6_device_max_running[] =
	{ 0, //host
		EnvironmentVariable<int>("NANOS6_CUDA_MAX", 32), //cuda
		0, //
		0, //
		EnvironmentVariable<int>("NANOS6_FPGA_MAX", 32) //FPGA
	};
	
	return nanos6_device_max_running[dev];
}

