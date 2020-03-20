/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef NULL_HARDWARE_COUNTERS_IMPLEMENTATION_HPP
#define NULL_HARDWARE_COUNTERS_IMPLEMENTATION_HPP

#include "hardware-counters/HardwareCountersInterface.hpp"
#include "hardware-counters/SupportedHardwareCounters.hpp"


class Task;

class NullHardwareCountersImplementation : public HardwareCountersInterface {

public:

	inline void initialize(bool, std::string)
	{
	}

	inline void shutdown()
	{
	}

	inline bool isSupported(HWCounters::counters_t)
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
		return 0.0;
	}

};

#endif // NULL_HARDWARE_COUNTERS_IMPLEMENTATION_HPP
