/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEFAULT_CPU_MANAGER_HPP
#define DEFAULT_CPU_MANAGER_HPP

#include "executors/threads/CPUManagerInterface.hpp"
#include "lowlevel/MultiConditionVariable.hpp"
#include "support/config/ConfigVariable.hpp"


class DefaultCPUManager : public CPUManagerInterface {

private:

	//! Identifies CPUs that are idle
	boost::dynamic_bitset<> _idleCPUs;

	//! Spinlock to access idle CPUs
	SpinLock _idleCPUsLock;

	//! The current number of idle CPUs, kept atomic through idleCPUsLock
	size_t _numIdleCPUs;

	//! The system ids of the CPUs in sponge mode. These CPUs are not used by
	//! the runtime to reduce the system noise
	ConfigVariableSet<size_t> _spongeModeCPUs;

	//! Condition variable where sponge CPUs (actually its running thread) is
	//! blocked until the runtime finalization
	MultiConditionVariable _spongeModeCondVar;

public:

	DefaultCPUManager() :
		CPUManagerInterface(),
		_spongeModeCPUs("cpumanager.sponge_cpus"),
		_spongeModeCondVar(_spongeModeCPUs.size())
	{
	}

	inline bool isSpongeCPU(CPU *cpu) const override
	{
		assert(cpu != nullptr);

		// Return whether the CPU is in the list of sponge CPUs
		return _spongeModeCPUs.contains(cpu->getSystemCPUId());
	}

	inline void enterSpongeMode(CPU *) override
	{
		_spongeModeCondVar.wait();
	}


	/*    CPUMANAGER    */

	void preinitialize() override;

	void initialize() override;

	void shutdownPhase1() override;

	inline void shutdownPhase2() override
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
	) override {
		assert(_cpuManagerPolicy != nullptr);

		_cpuManagerPolicy->execute(cpu, hint, numRequested);
	}

	inline CPU *getCPU(size_t systemCPUId) const override
	{
		assert(_systemToVirtualCPUId.size() > systemCPUId);

		size_t virtualCPUId = _systemToVirtualCPUId[systemCPUId];
		return _cpus[virtualCPUId];
	}

	inline CPU *getUnusedCPU() override
	{
		// In the default implementation, getting an unused CPU means going
		// through the idle mechanism and thus getting an idle CPU
		return getIdleCPU();
	}

	void forcefullyResumeFirstCPU() override;


	/*    CPUACTIVATION BRIDGE    */

	CPU::activation_status_t checkCPUStatusTransitions(WorkerThread *thread) override;

	inline void checkIfMustReturnCPU(WorkerThread *) override
	{
		// CPUs are never returned in this implementation
	}

	bool acceptsWork(CPU *cpu) override;

	bool enable(size_t systemCPUId) override;

	bool disable(size_t systemCPUId) override;


	/*    SHUTDOWN CPUS    */

	inline CPU *getShutdownCPU() override
	{
		// In the default implementation, getting a shutdown CPU means going
		// through the idle mechanism and thus getting an idle CPU
		return getIdleCPU();
	}

	inline void addShutdownCPU(CPU *cpu) override
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
