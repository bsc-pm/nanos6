/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PAPI_HARDWARE_COUNTERS_IMPLEMENTATION_HPP
#define PAPI_HARDWARE_COUNTERS_IMPLEMENTATION_HPP

#include "hardware-counters/HardwareCountersInterface.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


class Task;

class PAPIHardwareCountersImplementation : public HardwareCountersInterface {

public:

	inline void initialize(bool verbose, std::string verboseFile)
	{
		// TODO: Implement the PAPI backend
		FatalErrorHandler::failIf(true, "PAPI backend not supported yet");
	}

	inline void shutdown()
	{
	}

	inline bool isSupported(counters_t)
	{
		return false;
	}

	inline void threadInitialized()
	{
	}

	inline void threadShutdown()
	{
	}

	inline void taskCreated(Task *)
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

	inline size_t getTaskHardwareCountersSize()
	{
		return 0.0;
	}

};

#endif // PAPI_HARDWARE_COUNTERS_IMPLEMENTATION_HPP
