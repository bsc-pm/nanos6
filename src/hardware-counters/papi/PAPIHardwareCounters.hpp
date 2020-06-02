/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PAPI_HARDWARE_COUNTERS_HPP
#define PAPI_HARDWARE_COUNTERS_HPP

#include "hardware-counters/HardwareCountersInterface.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


class Task;

class PAPIHardwareCounters : public HardwareCountersInterface {

public:

	inline PAPIHardwareCounters(bool, const std::string &, const std::vector<bool> &)
	{
		FatalErrorHandler::fail("PAPI backend not supported yet");
	}

	inline ~PAPIHardwareCounters()
	{
	}

	inline void threadInitialized(ThreadHardwareCountersInterface *)
	{
	}

	inline void threadShutdown(ThreadHardwareCountersInterface *)
	{
	}

	inline void taskCreated(Task *, bool)
	{
	}

	inline void taskReinitialized(TaskHardwareCountersInterface *)
	{
	}

	inline void taskStarted(ThreadHardwareCountersInterface *, TaskHardwareCountersInterface *)
	{
	}

	inline void taskStopped(ThreadHardwareCountersInterface *, TaskHardwareCountersInterface *)
	{
	}

	inline void taskFinished(Task *, TaskHardwareCountersInterface *)
	{
	}

};

#endif // PAPI_HARDWARE_COUNTERS_HPP
