/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "IdlePolicy.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/cpu-managers/default/DefaultCPUManagerImplementation.hpp"


void IdlePolicy::execute(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numTasks)
{
	// NOTE: This policy works as follows:
	// - If the hint is IDLE_CANDIDATE, we try to idle the current CPU
	// - If the hint is ADDED_TASKS, we try to wake up as many idle CPUs
	//   as tasks were added
	// - If the hint is HANDLE_TASKFOR, we try to wake up all idle CPUs
	//   that can collaborate executing it
	if (hint == IDLE_CANDIDATE) {
		assert(cpu != nullptr);

		bool cpuIsIdle = DefaultCPUManagerImplementation::cpuBecomesIdle((CPU *) cpu);
		if (cpuIsIdle) {
			WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
			assert(currentThread != nullptr);

			ThreadManager::addIdler(currentThread);
			currentThread->switchTo(nullptr);

			// The thread may have migrated, update the compute place
			cpu = currentThread->getComputePlace();
			assert(cpu != nullptr);
		}
	} else if (hint == ADDED_TASKS) {
		assert(numTasks > 0);

		// At most we will obtain as many idle CPUs as the maximum amount
		size_t numCPUsToObtain = std::min(_numCPUs, numTasks);
		CPU *idleCPUs[numCPUsToObtain];

		// Try to get as many idle CPUs as we need
		size_t numCPUsObtained = DefaultCPUManagerImplementation::getIdleCPUs(
			numCPUsToObtain,
			idleCPUs
		);

		// Resume an idle thread for every idle CPU that has awakened
		for (size_t i = 0; i < numCPUsObtained; ++i) {
			assert(idleCPUs[i] != nullptr);
			ThreadManager::resumeIdle(idleCPUs[i]);
		}
	} else { // hint = HANDLE_TASKFOR
		assert(cpu != nullptr);

		std::vector<CPU *> idleCPUs;
		DefaultCPUManagerImplementation::getIdleCollaborators(idleCPUs, cpu);

		// Resume an idle thread for every unidled collaborator
		for (size_t i = 0; i < idleCPUs.size(); ++i) {
			assert(idleCPUs[i] != nullptr);
			ThreadManager::resumeIdle(idleCPUs[i]);
		}
	}
}
