#include "system/RuntimeInfo.hpp"

#include "SchedulerGenerator.hpp"
#include "Scheduler.hpp"
#include "SchedulerInterface.hpp"


SchedulerInterface *Scheduler::_scheduler;


void Scheduler::initialize()
{
	_scheduler = SchedulerGenerator::createHostScheduler();
	RuntimeInfo::addEntry("scheduler", "Scheduler", _scheduler->getName());
}

void Scheduler::shutdown() 
{
	delete _scheduler;
}


#include "instrument/support/InstrumentThreadLocalDataSupport.hpp"
#include "instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp"
