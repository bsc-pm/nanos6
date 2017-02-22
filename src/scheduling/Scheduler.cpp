#include "hierarchy/HostHierarchicalScheduler.hpp"
#include "Scheduler.hpp"
#include "SchedulerInterface.hpp"

#include "lowlevel/EnvironmentVariable.hpp"

#include <iostream>
#include <string>


SchedulerInterface *Scheduler::_scheduler;


void Scheduler::initialize()
{
	_scheduler = new HostHierarchicalScheduler();
}

void Scheduler::shutdown() 
{
	delete _scheduler;
}
