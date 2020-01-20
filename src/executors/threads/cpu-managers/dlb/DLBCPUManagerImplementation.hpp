/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef DLB_CPU_MANAGER_IMPLEMENTATION_HPP
#define DLB_CPU_MANAGER_IMPLEMENTATION_HPP

#include <cstring>
#include <sched.h>
#include <vector>

#include "executors/threads/CPU.hpp"
#include "executors/threads/CPUManagerInterface.hpp"
#include "hardware/places/ComputePlace.hpp"


class DLBCPUManagerImplementation : public CPUManagerInterface {

private:

	//! Default options (same as DLB_ARGS envvar)
	char _dlbOptions[64];


private:

	//! \brief Get a CPU set of all possible collaborators that can collaborate
	//! with a taskfor owned by a certain CPU
	//!
	//! \param[in] cpu The CPU that owns the taskfor
	//!
	//! \return A CPU set signaling which are its collaborators
	inline cpu_set_t getCollaboratorMask(CPU *cpu)
	{
		assert(cpu != nullptr);

		// The resulting mask of collaborators
		cpu_set_t resultMask;
		CPU_ZERO(&resultMask);

		CPU *candidate;
		size_t groupId = cpu->getGroupId();
		for (size_t id = 0; id < _cpus.size(); ++id) {
			candidate = _cpus[id];
			assert(candidate != nullptr);

			// A candidate is valid if it has the same group id as the cpu
			if (candidate->getGroupId() == groupId) {
				CPU_SET(candidate->getSystemCPUId(), &resultMask);
			}
		}

		return resultMask;
	}


//! NOTE: Documentation for methods available in CPUManagerInterface.hpp
public:

	inline DLBCPUManagerImplementation()
	{
		// Lend When Idle API mode
		strcpy(_dlbOptions, "--lewi --quiet=yes");
	}

	void preinitialize();

	void initialize();

	void shutdownPhase1();

	void shutdownPhase2();

	void executeCPUManagerPolicy(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numTasks = 0);

	inline CPU *getCPU(size_t systemCPUId) const
	{
		return _cpus[systemCPUId];
	}


	/*    CPUACTIVATION BRIDGE    */

	CPU::activation_status_t checkCPUStatusTransitions(WorkerThread *thread);

	bool acceptsWork(CPU *cpu);

	bool enable(size_t systemCPUId);

	bool disable(size_t systemCPUId);


	/*    IDLE CPUS    */

	// NOTE: By default, in this implementation we cancel the idle-CPU
	// mechanism by overriding it. If a CPU is idle we simply lend it using DLB

	bool cpuBecomesIdle(CPU *cpu, bool inShutdown = false);

	CPU *getIdleCPU(bool inShutdown = false);

	inline size_t getIdleCPUs(std::vector<CPU *> &, size_t)
	{
		return 0;
	}

	inline CPU *getIdleNUMANodeCPU(size_t)
	{
		return nullptr;
	}

	inline bool unidleCPU(CPU *)
	{
		return false;
	}

	inline void getIdleCollaborators(std::vector<CPU *> &, ComputePlace *)
	{
	}

	inline size_t getNumCPUsPerTaskforGroup() const
	{
		return HardwareInfo::getComputePlaceCount(nanos6_host_device) / _taskforGroups;
	}

};


#endif // DLB_CPU_MANAGER_IMPLEMENTATION_HPP
