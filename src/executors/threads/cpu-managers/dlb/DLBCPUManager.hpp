/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef DLB_CPU_MANAGER_HPP
#define DLB_CPU_MANAGER_HPP

#include <cstring>
#include <sched.h>
#include <vector>

#include "executors/threads/CPU.hpp"
#include "executors/threads/CPUManagerInterface.hpp"
#include "hardware/places/ComputePlace.hpp"


class DLBCPUManager : public CPUManagerInterface {

private:

	//! CPUs available to be used for shutdown purposes
	static boost::dynamic_bitset<> _shutdownCPUs;

	//! Spinlock to access shutdown CPUs
	static SpinLock _shutdownCPUsLock;

	//! A set of collaborator masks per CPU
	static std::vector<cpu_set_t> _collaboratorMasks;

public:

	/*    CPUMANAGER    */

	void preinitialize();

	void initialize();

	void shutdownPhase1();

	void shutdownPhase2();

	inline void executeCPUManagerPolicy(
		ComputePlace *cpu,
		CPUManagerPolicyHint hint,
		size_t numTasks = 0
	) {
		assert(_cpuManagerPolicy != nullptr);

		_cpuManagerPolicy->execute(cpu, hint, numTasks);
	}

	inline CPU *getCPU(size_t systemCPUId) const
	{
		return _cpus[systemCPUId];
	}

	inline CPU *getUnusedCPU()
	{
		// In the DLB implementation, underlying policies control CPUs,
		// obtaining unused CPUs should not be needed
		return nullptr;
	}

	void forcefullyResumeFirstCPU();


	/*    CPUACTIVATION BRIDGE    */

	CPU::activation_status_t checkCPUStatusTransitions(WorkerThread *thread);

	bool acceptsWork(CPU *cpu);

	bool enable(size_t systemCPUId);

	bool disable(size_t systemCPUId);


	/*    SHUTDOWN CPUS    */

	//! \brief Get an unused CPU to participate in the shutdown process
	//!
	//! \return A CPU or nullptr
	inline CPU *getShutdownCPU()
	{
		std::lock_guard<SpinLock> guard(_shutdownCPUsLock);

		boost::dynamic_bitset<>::size_type id = _shutdownCPUs.find_first();
		if (id != boost::dynamic_bitset<>::npos) {
			_shutdownCPUs[id] = false;
			return _cpus[id];
		} else {
			return nullptr;
		}
	}

	//! \brief Mark that a CPU is able to participate in the shutdown process
	//!
	//! \param[in] cpu The CPU object to offer
	inline void addShutdownCPU(CPU *cpu)
	{
		const int index = cpu->getIndex();

		_shutdownCPUsLock.lock();
		_shutdownCPUs[index] = true;
		_shutdownCPUsLock.unlock();
	}


	/*    DLB MECHANISM    */

	//! \brief Get a CPU set of all possible collaborators that can collaborate
	//! with a taskfor owned by a certain CPU
	//!
	//! \param[in,out] cpu The CPU that owns the taskfor
	//!
	//! \return A CPU set signaling which are its collaborators
	static inline cpu_set_t getCollaboratorMask(CPU *cpu)
	{
		assert(cpu != nullptr);

		// Return the mask of CPUs that can collaborate with 'cpu' in taskfors
		return _collaboratorMasks[cpu->getIndex()];
	}

};


#endif // DLB_CPU_MANAGER_HPP
