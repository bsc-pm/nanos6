#include "DefaultScheduler.hpp"
#include "FIFOImmediateSuccessorWithPollingScheduler.hpp"
#include "FIFOScheduler.hpp"
#include "ImmediateSuccessorScheduler.hpp"
#include "ImmediateSuccessorWithPollingScheduler.hpp"
#include "PriorityScheduler.hpp"
#include "Scheduler.hpp"
#include "SchedulerInterface.hpp"

#include "lowlevel/EnvironmentVariable.hpp"

#include <iostream>
#include <string>


SchedulerInterface *Scheduler::_scheduler;


void Scheduler::initialize()
{
	EnvironmentVariable<std::string> schedulerName("NANOS6_SCHEDULER", "default");
	
	if (schedulerName.getValue() == "default") {
		_scheduler = new DefaultScheduler();
	} else if (schedulerName.getValue() == "fifo") {
		_scheduler = new FIFOScheduler();
	} else if (schedulerName.getValue() == "immediatesuccessor") {
		_scheduler = new ImmediateSuccessorScheduler();
	} else if (schedulerName.getValue() == "iswp") {
		_scheduler = new ImmediateSuccessorWithPollingScheduler();
	} else if (schedulerName.getValue() == "fifoiswp") {
		_scheduler = new FIFOImmediateSuccessorWithPollingScheduler();
	} else if (schedulerName.getValue() == "priority") {
		_scheduler = new PriorityScheduler();
	} else {
		std::cerr << "Warning: invalid scheduler name '" << schedulerName.getValue() << "', using default instead." << std::endl;
		_scheduler = new DefaultScheduler();
	}
}

void Scheduler::shutdown() 
{
	delete _scheduler;
}


#include "instrument/support/InstrumentThreadLocalDataSupport.hpp"
#include "instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp"
