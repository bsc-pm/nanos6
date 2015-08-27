#include "CPU.hpp"

#include <cassert>

#include <pthread.h>
#include <unistd.h>


CPU::CPU(size_t systemCPUId, size_t virtualCPUId)
	: _activationStatus(starting_status), _systemCPUId(systemCPUId), _virtualCPUId(virtualCPUId), _shutdownControlerThread(nullptr)
{
	CPU_ZERO_S(sizeof(cpu_set_t), &_cpuMask);
	CPU_SET_S(systemCPUId, sizeof(cpu_set_t), &_cpuMask);
	
	int rc = pthread_attr_init(&_pthreadAttr);
	assert(rc == 0);
	
	rc = pthread_attr_setaffinity_np(&_pthreadAttr, sizeof(cpu_set_t), &_cpuMask);
	assert(rc == 0);
}

