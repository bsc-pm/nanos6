/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include "CPU.hpp"
#include "CPUThreadingModelData.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


CPU::CPU(size_t systemCPUId, size_t virtualCPUId, size_t NUMANodeId, L2Cache *l2Cache, L3Cache *l3Cache) :
	CPUPlace(virtualCPUId, l2Cache, l3Cache),
	_activationStatus(uninitialized_status),
	_systemCPUId(systemCPUId),
	_NUMANodeId(NUMANodeId),
	_hardwareCounters()
{
	CPU_ZERO(&_cpuMask);
	CPU_SET(systemCPUId, &_cpuMask);

	int rc = pthread_attr_init(&_pthreadAttr);
	FatalErrorHandler::handle(rc, " in call to pthread_attr_init");

	rc = pthread_attr_setstacksize(&_pthreadAttr, CPUThreadingModelData::getDefaultStackSize());
	FatalErrorHandler::handle(rc, " in call to pthread_attr_init");

	// Some machines, particularly ARM-based, do not always provide cache info.
	if (l2Cache != nullptr) {
		l2Cache->addCPU(this);
	}

	// L3Cache is not mandatory (e.g. KNL in flat mode has no L3)
	if (l3Cache != nullptr) {
		l3Cache->addCPU(this);
	}
}

