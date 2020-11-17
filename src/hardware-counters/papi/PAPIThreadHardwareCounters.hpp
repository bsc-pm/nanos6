/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PAPI_THREAD_HARDWARE_COUNTERS_HPP
#define PAPI_THREAD_HARDWARE_COUNTERS_HPP

#include <papi.h>

#include "hardware-counters/ThreadHardwareCountersInterface.hpp"


class PAPIThreadHardwareCounters : public ThreadHardwareCountersInterface {

private:

	//! The PAPI event set that must be read
	int _eventSet;

public:

	inline PAPIThreadHardwareCounters()
	{
		_eventSet = PAPI_NULL;
	}

	inline ~PAPIThreadHardwareCounters()
	{
	}

	inline int getEventSet() const
	{
		return _eventSet;
	}

	inline void setEventSet(int eventSet)
	{
		_eventSet = eventSet;
	}

};

#endif // PAPI_THREAD_HARDWARE_COUNTERS_HPP
