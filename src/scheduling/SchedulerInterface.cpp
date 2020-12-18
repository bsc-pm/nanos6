/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "Scheduler.hpp"
#include "SchedulerGenerator.hpp"
#include "executors/threads/CPUManager.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "support/config/ConfigVariable.hpp"
#include "system/RuntimeInfo.hpp"

ConfigVariable<std::string> SchedulerInterface::_schedulingPolicy("scheduler.policy");
ConfigVariable<bool> SchedulerInterface::_enableImmediateSuccessor("scheduler.immediate_successor");
ConfigVariable<bool> SchedulerInterface::_enablePriority("scheduler.priority");


SchedulerInterface::SchedulerInterface()
{
	SchedulingPolicy policy;
	if (_schedulingPolicy.getValue() == "fifo") {
		policy = FIFO_POLICY;
	} else if (_schedulingPolicy.getValue() == "lifo") {
		policy = LIFO_POLICY;
	} else {
		FatalErrorHandler::fail("Invalid scheduling policy ", _schedulingPolicy.getValue());
		return;
	}

	RuntimeInfo::addEntry("schedulingPolicy", "SchedulingPolicy", _schedulingPolicy);

	size_t computePlaceCount;
	computePlaceCount = CPUManager::getTotalCPUs();
	_hostScheduler = SchedulerGenerator::createHostScheduler(
		computePlaceCount, policy, _enablePriority,
		_enableImmediateSuccessor, _enableLocality);

	size_t totalDevices = (nanos6_device_t::nanos6_device_type_num);

	for (size_t i = 0; i < totalDevices; i++) {
		_deviceSchedulers[i] = nullptr;
	}

#if USE_CUDA
	computePlaceCount = HardwareInfo::getComputePlaceCount(nanos6_cuda_device);
	_deviceSchedulers[nanos6_cuda_device] =
		SchedulerGenerator::createDeviceScheduler(
			computePlaceCount, policy, _enablePriority,
			_enableImmediateSuccessor, nanos6_cuda_device);
#endif
#if USE_OPENACC
	computePlaceCount = HardwareInfo::getComputePlaceCount(nanos6_openacc_device);
	_deviceSchedulers[nanos6_openacc_device] =
		SchedulerGenerator::createDeviceScheduler(
			computePlaceCount, policy, _enablePriority,
			_enableImmediateSuccessor, nanos6_openacc_device);
#endif
#if NANOS6_OPENCL
	FatalErrorHandler::failIf(true, "OpenCL is not supported yet.");
#endif
#if USE_FPGA
	FatalErrorHandler::failIf(true, "FPGA is not supported yet.");
#endif

	_expiredTasks = 0;
}

SchedulerInterface::~SchedulerInterface()
{
	delete _hostScheduler;
#if USE_CUDA
	delete _deviceSchedulers[nanos6_cuda_device];
#endif
#if USE_OPENACC
	delete _deviceSchedulers[nanos6_openacc_device];
#endif
#if NANOS6_OPENCL
	FatalErrorHandler::failIf(true, "OpenCL is not supported yet.");
#endif
#if USE_FPGA
	FatalErrorHandler::failIf(true, "FPGA is not supported yet.");
#endif
	std::cout << "Expired tasks: " << _expiredTasks << std::endl;
}
