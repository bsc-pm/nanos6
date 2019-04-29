/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#include "ClusterScheduler.hpp"
#include "LocalScheduler.hpp"
#include "Scheduler.hpp"

#include <ClusterManager.hpp>

SchedulerInterface *Scheduler::_instance;

void Scheduler::initialize()
{
	bool clusterEnabled = ClusterManager::inClusterMode();
	if (clusterEnabled) {
		_instance = new ClusterScheduler();
	}
	else {
		_instance = new LocalScheduler();
	}
	
	assert(_instance != nullptr);
	RuntimeInfo::addEntry("scheduler", "Scheduler", _instance->getName());
}

void Scheduler::shutdown()
{
	delete _instance;
}
