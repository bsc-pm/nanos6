/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2021 Barcelona Supercomputing Center (BSC)
*/

#include "LocalScheduler.hpp"
#include "Scheduler.hpp"
#include "system/RuntimeInfo.hpp"

SchedulerInterface *Scheduler::_instance;

void Scheduler::initialize()
{
	_instance = new LocalScheduler();

	assert(_instance != nullptr);
	RuntimeInfo::addEntry("scheduler", "Scheduler", _instance->getName());
}

void Scheduler::shutdown()
{
	delete _instance;
}
