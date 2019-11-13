/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "DefaultCPUActivation.hpp"
#include "DefaultCPUManagerImplementation.hpp"


void DefaultCPUManagerImplementation::shutdownPhase1()
{
	// Notify all CPUs that the runtime is shutting down
	for (size_t id = 0; id < _cpus.size(); ++id) {
		if (_cpus[id] != nullptr) {
			DefaultCPUActivation::shutdownCPU(_cpus[id]);
		}
	}
}


/*    CPUACTIVATION BRIDGE    */

CPU::activation_status_t DefaultCPUManagerImplementation::checkCPUStatusTransitions(WorkerThread *thread)
{
	return DefaultCPUActivation::checkCPUStatusTransitions(thread);
}

bool DefaultCPUManagerImplementation::acceptsWork(CPU *cpu)
{
	return DefaultCPUActivation::acceptsWork(cpu);
}

bool DefaultCPUManagerImplementation::enable(size_t systemCPUId)
{
	return DefaultCPUActivation::enable(systemCPUId);
}

bool DefaultCPUManagerImplementation::disable(size_t systemCPUId)
{
	return DefaultCPUActivation::disable(systemCPUId);
}
