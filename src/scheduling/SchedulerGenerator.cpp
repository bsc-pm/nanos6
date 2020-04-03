/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include "SchedulerGenerator.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "scheduling/schedulers/HostScheduler.hpp"
#include "scheduling/schedulers/device/CUDADeviceScheduler.hpp"
#include "scheduling/schedulers/device/FPGADeviceScheduler.hpp"

HostScheduler *SchedulerGenerator::createHostScheduler(size_t totalComputePlaces, SchedulingPolicy policy, bool enablePriority, bool enableImmediateSuccessor)
{
	return new HostScheduler(totalComputePlaces, policy, enablePriority, enableImmediateSuccessor);
}

DeviceScheduler *SchedulerGenerator::createDeviceScheduler(size_t totalComputePlaces, SchedulingPolicy policy, bool enablePriority, bool enableImmediateSuccessor, nanos6_device_t deviceType)
{
	switch(deviceType) {
		case nanos6_cuda_device:
			return new CUDADeviceScheduler(totalComputePlaces, policy, enablePriority, enableImmediateSuccessor, deviceType);
		case nanos6_opencl_device:
			FatalErrorHandler::failIf(1, "OpenCL is not supported yet.");
			break;
		case nanos6_fpga_device:
			return new FPGADeviceScheduler(totalComputePlaces, policy, enablePriority, enableImmediateSuccessor, deviceType);
		case nanos6_cluster_device:
			FatalErrorHandler::failIf(1, "Cluster is not actually a device.");
			break;
		default:
			FatalErrorHandler::failIf(1, "Unknown or unsupported device.");
	}
	return nullptr;
}

