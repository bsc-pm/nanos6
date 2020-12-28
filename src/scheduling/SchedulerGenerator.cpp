/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2021 Barcelona Supercomputing Center (BSC)
*/

#include "SchedulerGenerator.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "scheduling/schedulers/HostScheduler.hpp"
#include "scheduling/schedulers/device/DeviceScheduler.hpp"

HostScheduler *SchedulerGenerator::createHostScheduler(
	size_t totalComputePlaces,
	SchedulingPolicy policy,
	bool enablePriority,
	bool enableImmediateSuccessor)
{
	return new HostScheduler(totalComputePlaces, policy, enablePriority, enableImmediateSuccessor);
}

DeviceScheduler *SchedulerGenerator::createDeviceScheduler(
	size_t totalComputePlaces,
	SchedulingPolicy policy,
	bool enablePriority,
	bool enableImmediateSuccessor,
	nanos6_device_t deviceType)
{
	switch(deviceType) {
		case nanos6_cuda_device:
			return new DeviceScheduler(
				totalComputePlaces,
				policy,
				enablePriority,
				enableImmediateSuccessor,
				deviceType,
				"CUDADeviceScheduler");
		case nanos6_openacc_device:
			return new DeviceScheduler(
				totalComputePlaces,
				policy,
				enablePriority,
				enableImmediateSuccessor,
				deviceType,
				"OpenAccDeviceScheduler");
		case nanos6_opencl_device:
			FatalErrorHandler::fail("OpenCL is not supported yet.");
			break;
		case nanos6_fpga_device:
			FatalErrorHandler::fail("FPGA is not supported yet.");
			break;
		case nanos6_cluster_device:
			FatalErrorHandler::fail("Cluster is not actually a device.");
			break;
		default:
			FatalErrorHandler::fail("Unknown or unsupported device.");
	}
	return nullptr;
}

