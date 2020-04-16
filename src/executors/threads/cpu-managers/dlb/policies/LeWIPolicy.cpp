/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "LeWIPolicy.hpp"
#include "executors/threads/cpu-managers/dlb/DLBCPUActivation.hpp"
#include "executors/threads/cpu-managers/dlb/DLBCPUManager.hpp"


void LeWIPolicy::execute(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numTasks)
{
	// NOTE This policy works as follows:
	// - If the hint is IDLE_CANDIDATE we try to lend the CPU if possible
	// - If the hint is ADDED_TASKS, we try to reclaim as many lent CPUs
	//   as tasks were added, or acquire new ones
	// - If the hint is HANDLE_TASKFOR, we try to reclaim all CPUs that can
	//   collaborate in the taskfor
	CPU *currentCPU = (CPU *) cpu;
	if (hint == IDLE_CANDIDATE) {
		assert(currentCPU != nullptr);

		if (currentCPU->isOwned()) {
			DLBCPUActivation::lendCPU(currentCPU);
		} else {
			DLBCPUActivation::returnCPU(currentCPU);
		}
	} else if (hint == ADDED_TASKS) {
		assert(numTasks > 0);

		// Try to obtain as many CPUs as tasks were added
		size_t numToObtain = std::min(_numCPUs, numTasks);
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
