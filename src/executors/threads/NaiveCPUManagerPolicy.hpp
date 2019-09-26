/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef NAIVE_CPU_MANAGER_POLICY_HPP
#define NAIVE_CPU_MANAGER_POLICY_HPP

#include <vector>

#include "CPU.hpp"
#include "CPUManager.hpp"
#include "CPUManagerPolicyInterface.hpp"
#include "ThreadManager.hpp"
#include "WorkerThread.hpp"

#include <InstrumentComputePlaceManagement.hpp>


class NaiveCPUManagerPolicy : public CPUManagerPolicyInterface {

public:
	
	//! NOTE: This policy works as follows:
	//! - CPUs are idled if the hint is IDLE_CANDIDATE and the runtime is not
	//!   shutting down
	//! - Idle CPUs are woken up if the hint is ADDED_TASKS
	//!   - Furthermore, as many CPUs are awaken as tasks are added (at most
	//!     the amount of idle CPUs)
	void executePolicy(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numTasks)
	{
		if (hint == IDLE_CANDIDATE) {
			assert(cpu != nullptr);
			
			WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
			assert(currentThread != nullptr);
			
			bool cpuIsIdle = CPUManager::cpuBecomesIdle((CPU *) cpu);
			if (cpuIsIdle) {
				// Account this CPU as idle and mark the thread as idle
				Instrument::suspendingComputePlace(cpu->getInstrumentationId());
				
				ThreadManager::addIdler(currentThread);
				currentThread->switchTo(nullptr);
				
				// The thread may have migrated, update the compute place
				cpu = currentThread->getComputePlace();
				assert(cpu != nullptr);
				
				Instrument::resumedComputePlace(cpu->getInstrumentationId());
			}
		} else { // hint = ADDED_TASKS
			// At most we will obtain as many idle CPUs as the maximum amount
			size_t numCPUsToObtain = std::min((size_t) CPUManager::getTotalCPUs(), numTasks);
			std::vector<CPU *> idleCPUs(numCPUsToObtain, nullptr);
			
			// Try to get as many idle CPUs as we need
			size_t numCPUsObtained = CPUManager::getIdleCPUs(idleCPUs, numCPUsToObtain);
			
			// Resume an idle thread for every idle CPU that has awakened
			for (size_t i = 0; i < numCPUsObtained; ++i) {
				assert(idleCPUs[i] != nullptr);
				ThreadManager::resumeIdle(idleCPUs[i]);
			}
		}
	}
	
};

#endif // NAIVE_CPU_MANAGER_POLICY_HPP
