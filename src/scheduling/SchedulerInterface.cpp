/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "Scheduler.hpp"
#include "SchedulerGenerator.hpp"
#include "executors/threads/CPUManager.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "system/RuntimeInfo.hpp"

SchedulerInterface::SchedulerInterface()
{
	const EnvironmentVariable<std::string> schedulingPolicy("NANOS6_SCHEDULING_POLICY", "fifo");
	RuntimeInfo::addEntry("schedulingPolicy", "SchedulingPolicy", schedulingPolicy.getValue());
	SchedulingPolicy policy = (schedulingPolicy.getValue() == "LIFO" || schedulingPolicy.getValue() == "lifo") ? LIFO_POLICY : FIFO_POLICY;
	
	const EnvironmentVariable<bool> enableImmediateSuccessor("NANOS6_IMMEDIATE_SUCCESSOR", "1");
	const EnvironmentVariable<bool> enablePriority("NANOS6_PRIORITY", "1");
	
	size_t totalCPUs = CPUManager::getTotalCPUs();
	_hostScheduler = SchedulerGenerator::createHostScheduler(totalCPUs, policy, enablePriority, enableImmediateSuccessor);
	
	size_t totalDevices = (nanos6_device_t::nanos6_device_type_num);
	
	assert(_deviceSchedulers != nullptr);
	
	for (size_t i = 0; i < totalDevices; i++) {
		_deviceSchedulers[i] = nullptr;
	}
	
	__attribute__((unused)) size_t computePlaceCount;
#if USE_CUDA
	computePlaceCount = HardwareInfo::getComputePlaceCount(nanos6_cuda_device);
	_deviceSchedulers[nanos6_cuda_device] = SchedulerGenerator::createDeviceScheduler(computePlaceCount, policy, enablePriority, enableImmediateSuccessor, nanos6_cuda_device);
#endif
#if NANOS6_OPENCL
	FatalErrorHandler::failIf(true, "OpenCL is not supported yet.");
#endif
#if USE_FPGA
	computePlaceCount = HardwareInfo::getComputePlaceCount(nanos6_fpga_device);
	_deviceScheduler[nanos6_fpga_device] = SchedulerGenerator::createDeviceScheduler(computePlaceCount, policy, enablePriority, enableImmediateSuccessor, nanos6_fpga_device);
#endif
}

SchedulerInterface::~SchedulerInterface()
{
	delete _hostScheduler;
#if USE_CUDA
	delete _deviceSchedulers[nanos6_cuda_device];
#endif
#if NANOS6_OPENCL
	FatalErrorHandler::failIf(true, "OpenCL is not supported yet.");
#endif
#if USE_FPGA
	delete _deviceSchedulers[nanos6_fpga_device];
#endif
}
