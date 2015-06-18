#include "DefaultScheduler.hpp"
#include "Scheduler.hpp"
#include "SchedulerInterface.hpp"


SchedulerInterface *Scheduler::_scheduler;


void Scheduler::initialize()
{
	_scheduler = new DefaultScheduler();
}

