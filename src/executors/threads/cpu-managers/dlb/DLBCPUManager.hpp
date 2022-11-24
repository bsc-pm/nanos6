/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2022 Barcelona Supercomputing Center (BSC)
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

public:

	/*    CPUMANAGER    */

	void preinitialize();

	void initialize();

	inline bool isDLBEnabled() const
	{
		return true;
	}

	void shutdownPhase1();

	void shutdownPhase2();

	inline void executeCPUManagerPolicy(
		ComputePlace *cpu,
		CPUManagerPolicyHint hint,
		size_t numRequested = 0
	) {
		assert(_cpuManagerPolicy != nullptr);

		_cpuManagerPolicy->execute(cpu, hint, numRequested);
	}

	inline CPU *getCPU(size_t systemCPUId) const
	{
		assert(_systemToVirtualCPUId.size() > systemCPUId);

		size_t virtualCPUId = _systemToVirtualCPUId[systemCPUId];
		return _cpus[virtualCPUId];
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

	void checkIfMustReturnCPU(WorkerThread *thread);

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
};


#endif // DLB_CPU_MANAGER_HPP
