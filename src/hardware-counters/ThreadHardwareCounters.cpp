/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "HardwareCounters.hpp"
#include "SupportedHardwareCounters.hpp"
#include "ThreadHardwareCounters.hpp"


void ThreadHardwareCounters::initialize()
{
#if HAVE_PAPI
	if (HardwareCounters::isBackendEnabled(HWCounters::PAPI_BACKEND)) {
		_papiCounters = new PAPIThreadHardwareCounters();
	}
#endif

#if HAVE_PQOS
	if (HardwareCounters::isBackendEnabled(HWCounters::PQOS_BACKEND)) {
		_pqosCounters = new PQoSThreadHardwareCounters();
	}
#endif
}

void ThreadHardwareCounters::shutdown()
{
#if HAVE_PAPI
	if (HardwareCounters::isBackendEnabled(HWCounters::PAPI_BACKEND)) {
		delete _papiCounters;
	}
#endif

#if HAVE_PQOS
	if (HardwareCounters::isBackendEnabled(HWCounters::PQOS_BACKEND)) {
		delete _pqosCounters;;
	}
#endif
}
