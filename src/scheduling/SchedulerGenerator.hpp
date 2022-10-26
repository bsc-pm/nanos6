/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef SCHEDULER_GENERATOR_HPP
#define SCHEDULER_GENERATOR_HPP

#include <nanos6/task-instantiation.h>

#include "scheduling/ReadyQueue.hpp"

class DeviceScheduler;
class HostScheduler;

class SchedulerGenerator {
public:
	static HostScheduler *createHostScheduler(
		size_t totalComputePlaces,
		SchedulingPolicy policy,
		bool enablePriority);

	static DeviceScheduler *createDeviceScheduler(
		size_t totalComputePlaces,
		SchedulingPolicy policy,
		bool enablePriority,
		nanos6_device_t deviceType);
};


#endif // SCHEDULER_GENERATOR_HPP
