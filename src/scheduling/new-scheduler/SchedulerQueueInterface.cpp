/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "SchedulerQueueInterface.hpp"
#include "queue/FIFOQueue.hpp"

SchedulerQueueInterface *SchedulerQueueInterface::initialize()
{
	EnvironmentVariable<std::string> queueName("NANOS6_SCHEDULER_QUEUE", "fifo");
	
	if (queueName.getValue() == "fifo") {
		return new FIFOQueue();
	} else {
		FatalErrorHandler::failIf(true, "Invalid scheduler queue name ", queueName.getValue());
		return nullptr;
	}
}
