/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef SCHEDULER_INTERFACE_HPP
#define SCHEDULER_INTERFACE_HPP

#include "hardware/places/ComputePlace.hpp"
#include "scheduling/schedulers/HostScheduler.hpp"
#include "scheduling/schedulers/device/CUDADeviceScheduler.hpp"
#include "scheduling/schedulers/device/DeviceScheduler.hpp"
#include "scheduling/schedulers/device/FPGADeviceScheduler.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include <HardwareCounters.hpp>
#include <InstrumentTaskStatus.hpp>
#include <Monitoring.hpp>

class SchedulerInterface {
	HostScheduler *_hostScheduler;
	DeviceScheduler *_deviceSchedulers[nanos6_device_type_num];
	
public:
	SchedulerInterface();
	virtual ~SchedulerInterface();
	
	virtual inline void addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint = NO_HINT)
	{
		nanos6_device_t taskType = (nanos6_device_t) task->getDeviceType();
		assert(taskType != nanos6_cluster_device);
		
		if (taskType == nanos6_host_device) {
			_hostScheduler->addReadyTask(task, computePlace, hint);
		} else {
			assert(taskType == _deviceSchedulers[taskType]->getDeviceType());
			_deviceSchedulers[taskType]->addReadyTask(task, computePlace, hint);
		}
	}
	
	virtual inline Task *getReadyTask(ComputePlace *computePlace, ComputePlace *deviceComputePlace = nullptr)
	{
		assert(computePlace->getType() == nanos6_host_device);
		nanos6_device_t computePlaceType = (deviceComputePlace == nullptr) ? nanos6_host_device : deviceComputePlace->getType();
		
		if (computePlaceType == nanos6_host_device) {
			return _hostScheduler->getReadyTask(computePlace);
		} else {
			assert(deviceComputePlace->getType() != nanos6_cluster_device);
			return _deviceSchedulers[computePlaceType]->getReadyTask(computePlace, deviceComputePlace);
		}
	}
	
	//! \brief Get the amount of ready tasks in the queue
	//!
	//! \param[in] computePlace The host compute place
	//! \param[in] deviceComputePlace The target device compute place if it exists
	//!
	//! \returns The current amount of tasks in the ready queue
	virtual size_t getNumReadyTasks(__attribute__((unused)) ComputePlace *computePlace, ComputePlace *deviceComputePlace = nullptr)
	{
		assert(computePlace != nullptr);
		assert(computePlace->getType() == nanos6_host_device);
		nanos6_device_t computePlaceType = (deviceComputePlace == nullptr) ?
			nanos6_host_device : deviceComputePlace->getType();
		
		if (computePlaceType == nanos6_host_device) {
			return _hostScheduler->getNumReadyTasks();
		} else {
			assert(deviceComputePlace->getType() != nanos6_cluster_device);
			return _deviceSchedulers[computePlaceType]->getNumReadyTasks();
		}
	}
	
	virtual std::string getName() const = 0;
};

#endif // SCHEDULER_INTERFACE_HPP
