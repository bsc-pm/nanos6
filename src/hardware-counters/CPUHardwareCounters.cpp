/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "CPUHardwareCounters.hpp"
#include "HardwareCounters.hpp"

CPUHardwareCounters::CPUHardwareCounters()
{
#if HAVE_PAPI
	if (HardwareCounters::isBackendEnabled(HWCounters::PAPI_BACKEND)) {
		_papiCounters = new PAPICPUHardwareCounters();
	}
#endif

#if HAVE_PQOS
	if (HardwareCounters::isBackendEnabled(HWCounters::PQOS_BACKEND)) {
		_pqosCounters = new PQoSCPUHardwareCounters();
	}
#endif
}
