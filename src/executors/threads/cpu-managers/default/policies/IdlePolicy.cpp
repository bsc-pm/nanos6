/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2023 Barcelona Supercomputing Center (BSC)
*/

#include "IdlePolicy.hpp"
#include "executors/threads/ThreadManager.hpp"


void IdlePolicy::execute(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numRequested)
{
	// NOTE: This policy works as follows:
	// - If the hint is IDLE_CANDIDATE, we try to idle the current CPU
	// - If the hint is REQUEST_CPUS, we try to wake up the requested
	//   number of idle CPUs
	if (hint == IDLE_CANDIDATE) {
		assert(cpu != nullptr);

		bool cpuIsIdle = _cpuManager.cpuBecomesIdle((CPU *) cpu);
		if (cpuIsIdle) {
			WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
			assert(currentThread != nullptr);

			// Calls from the Instrument and Monitoring modules can be found within
			// the "cpuBecomesIdle" function, before releasing the CPU lock

			ThreadManager::addIdler(currentThread);
			currentThread->switchTo(nullptr);
		}
	} else { // hint == REQUEST_CPUS
		assert(numRequested > 0);

		// At most we will obtain as many idle CPUs as the maximum amount
		size_t numCPUsToObtain = std::min(_numCPUs, numRequested);
		CPU *idleCPUs[numCPUsToObtain];

		// Try to get as many idle CPUs as we need
		size_t numCPUsObtained = _cpuManager.getIdleCPUs(
			numCPUsToObtain,
			idleCPUs
		);

		// Resume an idle thread for every idle CPU that has awakened
		for (size_t i = 0; i < numCPUsObtained; ++i) {
			assert(idleCPUs[i] != nullptr);
			ThreadManager::resumeIdle(idleCPUs[i]);
		}
	}
}
