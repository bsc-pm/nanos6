/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include "IdlePolicy.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/cpu-managers/default/DefaultCPUManager.hpp"


void IdlePolicy::execute(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numRequested)
{
	// NOTE: This policy works as follows:
	// - If the hint is IDLE_CANDIDATE, we try to idle the current CPU
	// - If the hint is REQUEST_CPUS, we try to wake up the requested
	//   number of idle CPUs
	// - If the hint is HANDLE_TASKFOR, we try to wake up all idle CPUs
	//   that can collaborate executing it
	if (hint == IDLE_CANDIDATE) {
		assert(cpu != nullptr);

		bool cpuIsIdle = DefaultCPUManager::cpuBecomesIdle((CPU *) cpu);
		if (cpuIsIdle) {
			WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
			assert(currentThread != nullptr);

			// Calls from the Instrument and Monitoring modules can be found within
			// the "cpuBecomesIdle" function, before releasing the CPU lock

			ThreadManager::addIdler(currentThread);
			currentThread->switchTo(nullptr);
		}
	} else if (hint == REQUEST_CPUS) {
		assert(numRequested > 0);

		// At most we will obtain as many idle CPUs as the maximum amount
		size_t numCPUsToObtain = std::min(_numCPUs, numRequested);
		CPU *idleCPUs[numCPUsToObtain];

		// Try to get as many idle CPUs as we need
		size_t numCPUsObtained = DefaultCPUManager::getIdleCPUs(
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
		DefaultCPUManager::getIdleCollaborators(idleCPUs, cpu);

		// Resume an idle thread for every unidled collaborator
		for (size_t i = 0; i < idleCPUs.size(); ++i) {
			assert(idleCPUs[i] != nullptr);
			ThreadManager::resumeIdle(idleCPUs[i]);
		}
	}
}
