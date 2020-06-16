/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "GreedyPolicy.hpp"
#include "executors/threads/cpu-managers/dlb/DLBCPUActivation.hpp"
#include "executors/threads/cpu-managers/dlb/DLBCPUManager.hpp"


void GreedyPolicy::execute(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numRequested)
{
	// NOTE This policy works as follows:
	// - If the hint is IDLE_CANDIDATE we do not lend the CPU if it is owned
	//   We only lend CPUs if DLB asks for it (disable callback DLBCPUActivation)
	// - If the hint is REQUEST_CPUS, we try to reclaim the requested number of
	//   CPUs or acquire new ones
	// - If the hint is HANDLE_TASKFOR, we try to reclaim all CPUs that can
	//   collaborate in the taskfor
	CPU *currentCPU = (CPU *) cpu;
	if (hint == IDLE_CANDIDATE) {
		assert(currentCPU != nullptr);

		if (!currentCPU->isOwned()) {
			DLBCPUActivation::returnCPU(currentCPU);
		}
	} else if (hint == REQUEST_CPUS) {
		assert(numRequested > 0);

		// Try to obtain the requested number of CPUs
		size_t numToObtain = std::min(_numCPUs, numRequested);
		DLBCPUActivation::acquireCPUs(numToObtain);
	} else { // hint = HANDLE_TASKFOR
		assert(currentCPU != nullptr);

		// Try to reclaim any lent collaborator of the taskfor
		cpu_set_t cpuMask = DLBCPUManager::getCollaboratorMask(currentCPU);
		if (CPU_COUNT(&cpuMask) > 0) {
			DLBCPUActivation::acquireCPUs(cpuMask);
		}
	}
}
