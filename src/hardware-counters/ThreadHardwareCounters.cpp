/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "HardwareCounters.hpp"
#include "SupportedHardwareCounters.hpp"
#include "ThreadHardwareCounters.hpp"


void ThreadHardwareCounters::initialize()
{
	// NOTE: Objects are constructed in this function, but they are freed
	// each by their backend, respectively (see PQoSHardwareCounters.cpp::threadShutdown)

#if HAVE_PAPI
	if (HardwareCounters::isEnabled(HWCounters::PAPI_BACKEND)) {
		_papiCounters = new PAPIThreadHardwareCounters();
	}
#endif

#if HAVE_PQOS
	if (HardwareCounters::isEnabled(HWCounters::PQOS_BACKEND)) {
		_pqosCounters = new PQoSThreadHardwareCounters();
	}
#endif
}
