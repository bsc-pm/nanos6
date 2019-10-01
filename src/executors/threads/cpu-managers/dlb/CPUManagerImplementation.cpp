/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <dlb.h>

#include "CPUActivation.hpp"
#include "CPUManagerImplementation.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

#include <InstrumentComputePlaceManagement.hpp>
#include <Monitoring.hpp>


void CPUManagerImplementation::preinitialize()
{
	// First preinitialize using the CPUManagerInterface
	CPUManagerInterface::preinitialize();
	
	// Now initialize the DLB library
	// NOTE: We use the sync (or polling) version of the library. This means
	// that when a call to DLB returns, all the required actions have been
	// taken (i.e. all the callbacks have been triggered before returning)
	assert(_cpus.size() > 0);
	int ret = DLB_Init(_cpus.size(), &_cpuMask, _dlbOptions);
	FatalErrorHandler::failIf(
		ret != DLB_SUCCESS,
		"Error code ", ret, " while initializing DLB"
	);
	
	// Prepare callbacks to lend/reclaim CPUs
	ret = DLB_CallbackSet(
		dlb_callback_enable_cpu,
		(dlb_callback_t)CPUActivation::dlbEnableCallback,
		nullptr
	);
	FatalErrorHandler::failIf(
		ret != DLB_SUCCESS,
		"Error code ", ret, " while registering DLB callbacks"
	);
}

void CPUManagerImplementation::shutdownPhase1()
{
	// Make sure all CPUs are reclaimed
	CPU *cpu;
	CPU::activation_status_t status;
	for (size_t id = 0; id < _cpus.size(); ++id) {
		cpu = _cpus[id];
		if (cpu != nullptr) {
			status = cpu->getActivationStatus();
			
			// Try to reclaim it again until we finally have reclaimed it
			while (status != CPU::shutdown_status) {
				CPUActivation::shutdownCPU(cpu);
				status = cpu->getActivationStatus();
			}
		}
	}
}

void CPUManagerImplementation::shutdownPhase2()
{
	// Shutdown DLB
	// ret != DLB_SUCCESS means it was not initialized (should never occur)
	__attribute__((unused)) int ret = DLB_Finalize();
	assert(ret == DLB_SUCCESS);
}

void CPUManagerImplementation::executeCPUManagerPolicy(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numTasks)
{
	// NOTE This policy works as follows:
	// - If the hint is IDLE_CANDIDATE we try to lend the CPU if possible
	// - If the hint is ADDED_TASKS, we try to reclaim as many lent CPUs
	//   as tasks were added
	// - If the hint is ADDED_TASKFOR, we try to reclaim all CPUs that can
	//   collaborate in the taskfor
	if (hint == IDLE_CANDIDATE) {
		assert(cpu != nullptr);
		
		// Lend the CPU
		CPUActivation::lendCPU((CPU *) cpu);
	} else if (hint == ADDED_TASKS) {
		assert(numTasks > 0);
		
		// At most we will obtain as many lent CPUs as _cpus.size()
		size_t numToObtain = std::min(_cpus.size(), numTasks);
		CPUActivation::reclaimCPUs(numToObtain);
	} else { // hint = HANDLE_TASKFOR
		assert(cpu != nullptr);
		
		// Try to reclaim any lent collaborator of the taskfor
		cpu_set_t collaboratorMask = getCollaboratorMask((CPU *) cpu);
		CPUActivation::reclaimCPUs(collaboratorMask);
	}
}


// NOTE: The following functions should only be used when the runtime is
// shutting down, to let all threads have an oportunity to shutdown

bool CPUManagerImplementation::cpuBecomesIdle(CPU *cpu, bool inShutdown)
{
	if (inShutdown) {
		const int index = cpu->getIndex();
		_idleCPUsLock.lock();
		
		// The CPU should not be marked as idle
		assert(!_idleCPUs[index]);
		
		_idleCPUs[index] = true;
		_idleCPUsLock.unlock();
	}
	
	return inShutdown;
}

CPU *CPUManagerImplementation::getIdleCPU(bool inShutdown)
{
	if (inShutdown) {
		_idleCPUsLock.lock();
		boost::dynamic_bitset<>::size_type idleCPU = _idleCPUs.find_first();
		if (idleCPU != boost::dynamic_bitset<>::npos) {
			_idleCPUs[idleCPU] = false;
			_idleCPUsLock.unlock();
			
			return _cpus[idleCPU];
		}
		_idleCPUsLock.unlock();
	}
	
	return nullptr;
}

