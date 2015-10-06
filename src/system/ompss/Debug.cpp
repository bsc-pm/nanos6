#include "api/nanos6_debug_interface.h"
#include "executors/threads/ThreadManagerDebuggingInterface.hpp"


long nanos_get_num_cpus()
{
	static long activeCPUs = 0;
	
	if (activeCPUs == 0) {
		ThreadManagerDebuggingInterface::cpu_list_t &cpuList = ThreadManagerDebuggingInterface::getCPUListRef();
		cpu_set_t const &cpuMask = ThreadManagerDebuggingInterface::getProcessCPUMaskRef();
		for (size_t systemCPUId = 0; systemCPUId < CPU_SETSIZE; systemCPUId++) {
			if (CPU_ISSET(systemCPUId, &cpuMask)) {
				activeCPUs++;
			}
		}
	}
	
	return activeCPUs;
}

