#include "DefaultScheduler.hpp"
#include "FIFOScheduler.hpp"
#include "ImmediateSuccessorScheduler.hpp"
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
	} else {
		std::cerr << "Warning: invalid scheduler name '" << schedulerName.getValue() << "', using default instead." << std::endl;
		_scheduler = new DefaultScheduler();
	}
}

