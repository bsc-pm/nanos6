/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "HardwareCounters.hpp"
#include "TaskHardwareCounters.hpp"


void TaskHardwareCounters::initialize()
{
	// NOTE: Objects are constructed in this function, but they are freed when
	// the task is freed (see TaskFinalizationImplementation.hpp)
	assert(_allocationAddress != nullptr);

	// Use a copy since we may need the original allocation address
	void *currentAddress = _allocationAddress;
	if (HardwareCounters::isEnabled(HWCounters::PAPI_BACKEND)) {
#if HAVE_PAPI
		_papiCounters = new (currentAddress) PAPITaskHardwareCounters();
		currentAddress = (char *) currentAddress + sizeof(PAPITaskHardwareCounters);
#endif
	}

	if (HardwareCounters::isEnabled(HWCounters::PQOS_BACKEND)) {
#if HAVE_PQOS
		_pqosCounters = new (currentAddress) PQoSTaskHardwareCounters();
		currentAddress = (char *) currentAddress + sizeof(PQoSTaskHardwareCounters);
#endif
	}
}

size_t TaskHardwareCounters::getTaskHardwareCountersSize()
{
	size_t totalSize = 0;

	if (HardwareCounters::isEnabled(HWCounters::PAPI_BACKEND)) {
#if HAVE_PAPI
		totalSize += sizeof(PAPITaskHardwareCounters);
#endif
	}

	if (HardwareCounters::isEnabled(HWCounters::PQOS_BACKEND)) {
#if HAVE_PQOS
		totalSize += sizeof(PQoSTaskHardwareCounters);
#endif
	}

	return totalSize;
}
