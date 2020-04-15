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

	inline PAPIHardwareCounters(bool, const std::string &)
	{
		FatalErrorHandler::failIf(true, "PAPI backend not supported yet");
	}

	inline ~PAPIHardwareCounters()
	{
	}

	inline bool isSupported(counters_t) const
	{
		return false;
	}

	inline void threadInitialized()
	{
	}

	inline void threadShutdown()
	{
	}

	inline void taskCreated(Task *, bool)
	{
	}

	inline void taskReinitialized(Task *)
	{
	}

	inline void taskStarted(Task *)
	{
	}

	inline void taskStopped(Task *)
	{
	}

	inline void taskFinished(Task *)
	{
	}

	inline size_t getTaskHardwareCountersSize() const
	{
		return 0;
	}

};

#endif // PAPI_HARDWARE_COUNTERS_HPP
