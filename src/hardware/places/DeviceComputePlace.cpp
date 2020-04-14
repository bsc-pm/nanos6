/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include <string>

#include <nanos6/polling.h>

#include "DeviceComputePlace.hpp"
#include "executors/threads/TaskFinalization.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/HardwareInfo.hpp"
#include "hardware/device/DeviceInfoImplementation.hpp"
#include "hardware/places/DeviceMemoryPlace.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/TaskImplementation.hpp"

#include <DataAccessRegistration.hpp>
#include <ExecutionWorkflow.hpp>

DeviceComputePlace::DeviceComputePlace(DeviceMemoryPlace *memoryPlace,
	nanos6_device_t type, int subType, int index, DeviceFunctionsInterface *functions, void *deviceHandler) :
	ComputePlace(index, type),
	_type(type),
	_subType(subType),
	_deviceHandler(deviceHandler),
	_maxRunningTasks(deviceMaxRunningTask(type)),
	_runningTasks(0),
	_strPollingService(std::string("runner:") + std::to_string(type) + " " + std::to_string(index) + " " + std::to_string(subType)),
	_functions(functions),
	_memoryPlace(memoryPlace)
{
}

DeviceComputePlace::~DeviceComputePlace()
{
	deactivatePollingService();
}

int DeviceComputePlace::pollingFinishTasks(DeviceFunctionsInterface *functions)
{
	std::vector<Task *> finishedTasks;
	functions->getFinishedTasks(finishedTasks);

	CPU *cpu = nullptr;
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	if (currentThread != nullptr) {
		cpu = currentThread->getComputePlace();
	}

	if (!finishedTasks.empty()) {
		CPUDependencyData localHpDependencyData;
		for (Task *task : finishedTasks) {
			// If 0 tasks must put their parent in the scheduler, because it is
			// blocked on If0Task::waitForIf0Task until this task ends.
			// Note that TaskFinalization::disposeTask will not unlock the
			// parent in this case.
			if (task->isIf0()) {
				Task *parent = task->getParent();
				assert(parent != nullptr);

				// Unlock parent that was waiting for this if0
				Scheduler::addReadyTask(parent, cpu, UNBLOCKED_TASK_HINT);

				// After adding a task, the CPUManager may want to unidle CPUs
				CPUManager::executeCPUManagerPolicy(cpu, ADDED_TASKS, 1);
			}

			if (task->markAsFinished(cpu)) {
				DataAccessRegistration::unregisterTaskDataAccesses(task, cpu,
					localHpDependencyData, task->getMemoryPlace(), true);

				Monitoring::taskFinished(task);
				HardwareCounters::taskFinished(task);
				TaskFinalization::taskFinished(task, cpu);

				if (task->markAsReleased()) {
					TaskFinalization::disposeTask(task);
				}
			}
		}
	}

	return false;
}

int DeviceComputePlace::pollingRun(DeviceComputePlace *deviceComputePlace)
{
	int i = 0;
	while (deviceComputePlace->canRunTask()) {
		if (i++ > deviceComputePlace->getMaxRunningTasks())
			return false;

		Task *task = Scheduler::getReadyTask(deviceComputePlace);

		if (task == nullptr)
			continue;

		task->setComputePlace(deviceComputePlace);
		deviceComputePlace->runTask(task);
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
	task->setMemoryPlace((MemoryPlace *)_memoryPlace);
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
		(nanos6_polling_service_t)pollingRun, this);
}

void DeviceComputePlace::activatePollingService()
{
	nanos6_register_polling_service(_strPollingService.c_str(),
		(nanos6_polling_service_t)pollingRun, this);
}

int DeviceComputePlace::deviceMaxRunningTask(nanos6_device_t dev)
{
	static int nanos6_device_max_running[] =
		{
			0,												 // Host
			EnvironmentVariable<int>("NANOS6_CUDA_MAX", 32), // CUDA
			0,												 //
			0,												 //
			EnvironmentVariable<int>("NANOS6_FPGA_MAX", 32)  // FPGA
		};

	return nanos6_device_max_running[dev];
}
