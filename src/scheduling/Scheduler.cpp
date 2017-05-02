#include "SchedulerGenerator.hpp"
#include "Scheduler.hpp"
#include "SchedulerInterface.hpp"


SchedulerInterface *Scheduler::_scheduler;


void Scheduler::initialize()
{
	_scheduler = SchedulerGenerator::createHostScheduler();
}

void Scheduler::shutdown() 
{
	delete _scheduler;
}


#include "instrument/support/InstrumentThreadLocalDataSupport.hpp"
#include "instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp"
