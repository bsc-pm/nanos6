#include "CPU.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

#include <pthread.h>
#include <unistd.h>


CPU::CPU(size_t systemCPUId, size_t virtualCPUId)
	: _activationStatus(uninitialized_status), _systemCPUId(systemCPUId), _virtualCPUId(virtualCPUId), _shutdownControlerThread(nullptr),
	_dependencyData()
{
	CPU_ZERO_S(sizeof(cpu_set_t), &_cpuMask);
	CPU_SET_S(systemCPUId, sizeof(cpu_set_t), &_cpuMask);
	
	int rc = pthread_attr_init(&_pthreadAttr);
	FatalErrorHandler::handle(rc, " in call to pthread_attr_init");
}

