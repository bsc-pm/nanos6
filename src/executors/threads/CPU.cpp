/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "CPU.hpp"
#include "CPUThreadingModelData.hpp"

#include "lowlevel/FatalErrorHandler.hpp"

#include <pthread.h>
#include <unistd.h>


CPU::CPU(size_t systemCPUId, size_t virtualCPUId, size_t NUMANodeId)
	: _activationStatus(uninitialized_status), _systemCPUId(systemCPUId), _virtualCPUId(virtualCPUId), _NUMANodeId(NUMANodeId)
{
	CPU_ZERO_S(sizeof(cpu_set_t), &_cpuMask);
	CPU_SET_S(systemCPUId, sizeof(cpu_set_t), &_cpuMask);
	
	int rc = pthread_attr_init(&_pthreadAttr);
	FatalErrorHandler::handle(rc, " in call to pthread_attr_init");
	
	rc = pthread_attr_setstacksize(&_pthreadAttr, CPUThreadingModelData::getDefaultStackSize());
	FatalErrorHandler::handle(rc, " in call to pthread_attr_init");
}

