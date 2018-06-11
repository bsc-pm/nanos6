/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef SCHEDULER_GENERATOR_HPP
#define SCHEDULER_GENERATOR_HPP

#include "SchedulerInterface.hpp"
#include "lowlevel/EnvironmentVariable.hpp"

#include <nanos6/task-instantiation.h>

class SchedulerGenerator {
private:
	// The hierarchical scheduler may be collapsable. In this case, if a node
	// from the hierarchy has only one children, it won't be generated.
	static bool _collapsable;
	
	// Get the CPU scheduler
	static SchedulerInterface *createCPUScheduler(std::string const &schedulerName, int nodeIndex);
	
	// Get the CUDA scheduler
	static SchedulerInterface *createCUDAScheduler(std::string const &schedulerName, int nodeIndex);
	
public:
	// Get the Host scheduler
	// This is the scheduler that is called through the Scheduler class. Therefor, this is the initializer
	static SchedulerInterface *createHostScheduler();
	
	// Get the scheduler for the NUMA nodes
	static SchedulerInterface *createNUMAScheduler();
	
	static SchedulerInterface *createNUMANodeScheduler(int nodeIndex);
	
	static SchedulerInterface *createDeviceScheduler(int nodeIndex, nanos6_device_t type);
};


#endif // SCHEDULER_GENERATOR_HPP
