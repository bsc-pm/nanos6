/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEFAULT_CPU_MANAGER_HPP
#define DEFAULT_CPU_MANAGER_HPP

#include "executors/threads/CPUManagerInterface.hpp"


class DefaultCPUManager : public CPUManagerInterface {

private:

	//! Identifies CPUs that are idle
	boost::dynamic_bitset<> _idleCPUs;

	//! Spinlock to access idle CPUs
	SpinLock _idleCPUsLock;

	//! The current number of idle CPUs, kept atomic through idleCPUsLock
	size_t _numIdleCPUs;

public:

	DefaultCPUManager() : CPUManagerInterface()
	{
	}

	/*    CPUMANAGER    */

	void preinitialize();

	void initialize();

	void shutdownPhase1();

	inline void shutdownPhase2()
	{
		delete _leaderThreadCPU;

		delete _cpuManagerPolicy;

		// Make sure the policy is nullptr to trip asserts if something's wrong
		_cpuManagerPolicy = nullptr;
	}

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
		// In the default implementation, getting an unused CPU means going
		// through the idle mechanism and thus getting an idle CPU
		return getIdleCPU();
	}

	void forcefullyResumeFirstCPU();


	/*    CPUACTIVATION BRIDGE    */

	CPU::activation_status_t checkCPUStatusTransitions(WorkerThread *thread);

	inline void checkIfMustReturnCPU(WorkerThread *)
	{
		// CPUs are never returned in this implementation
	}

	bool acceptsWork(CPU *cpu);

	bool enable(size_t systemCPUId);

	bool disable(size_t systemCPUId);


	/*    SHUTDOWN CPUS    */

	inline CPU *getShutdownCPU()
	{
		// In the default implementation, getting a shutdown CPU means going
		// through the idle mechanism and thus getting an idle CPU
		return getIdleCPU();
	}

	inline void addShutdownCPU(CPU *cpu)
	{
		assert(cpu != nullptr);

		// In the default implementation, adding a shutdown CPU means going
		// through the idle mechanism and thus adding an idle CPU
		const int index = cpu->getIndex();

		_idleCPUsLock.lock();
		_idleCPUs[index] = true;
		++_numIdleCPUs;
		assert(_numIdleCPUs <= _cpus.size());
		_idleCPUsLock.unlock();
	}


	/*    IDLE MECHANISM    */

	//! \brief Mark a CPU as idle
	//!
	//! \param[in,out] cpu The CPU object to idle
	//!
	//! \return Whether the operation was successful
	bool cpuBecomesIdle(CPU *cpu);

	//! \brief Try to get any idle CPU
	//!
	//! \return A CPU or nullptr
	CPU *getIdleCPU();

	//! \brief Get a specific number of idle CPUs
	//!
	//! \param[in] numCPUs The amount of CPUs to retreive
	//! \param[out] idleCPUs An array of at least size 'numCPUs' where the
	//! retreived idle CPUs will be placed
	//!
	//! \return The number of idle CPUs obtained/valid references in the vector
	size_t getIdleCPUs(size_t numCPUs, CPU *idleCPUs[]);
};

#endif // DEFAULT_CPU_MANAGER_HPP
