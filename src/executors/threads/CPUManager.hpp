/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_MANAGER_HPP
#define CPU_MANAGER_HPP

#include <config.h>

#include "CPUManagerInterface.hpp"
#include "executors/threads/cpu-managers/default/DefaultCPUManagerImplementation.hpp"
#if HAVE_DLB
#include "executors/threads/cpu-managers/dlb/DLBCPUManagerImplementation.hpp"
#endif
#include "lowlevel/EnvironmentVariable.hpp"


class CPUManager {

private:

	//! The CPU Manager instance
	static CPUManagerInterface *_cpuManager;

	//! Whether DLB is enabled
	static EnvironmentVariable<bool> _dlbEnabled;

public:

	/*    CPU MANAGER    */

	//! \brief Pre-initialize structures for the CPUManager
	static inline void preinitialize()
	{
		assert(_cpuManager == nullptr);

#if HAVE_DLB
		if (_dlbEnabled) {
			_cpuManager = new DLBCPUManagerImplementation();
		} else {
			_cpuManager = new DefaultCPUManagerImplementation();
		}
#else
		_cpuManager = new DefaultCPUManagerImplementation();
#endif
		assert(_cpuManager != nullptr);

		_cpuManager->preinitialize();
	}

	//! \brief Initialize all structures for the CPUManager
	static inline void initialize()
	{
		assert(_cpuManager != nullptr);

		_cpuManager->initialize();
	}

	//! \brief Check if CPU initialization has finished
	//!
	//! \return Whether initialization has finished
	static inline bool hasFinishedInitialization()
	{
		assert(_cpuManager != nullptr);

		return _cpuManager->hasFinishedInitialization();
	}

	//! \brief In the first phase of the shutdown, all CPUs are notified about
	//! the shutdown so that they are all available for the shutdown process
	static inline void shutdownPhase1()
	{
		assert(_cpuManager != nullptr);

		_cpuManager->shutdownPhase1();
	}

	//! \brief In the second phase of the shutdown all needed CPU-related
	//! structures are freed
	static inline void shutdownPhase2()
	{
		assert(_cpuManager != nullptr);

		_cpuManager->shutdownPhase2();

		delete _cpuManager;
	}

	//! \brief This method is executed after the amount of work in the runtime
	//! changes. Some common scenarios include:
	//! - Adding tasks in the scheduler (hint = ADDED_TASKS)
	//! - Execution of a taskfor (hint = HANDLE_TASKFOR)
	//! - Running out of tasks to execute (hint = IDLE_CANDIDATE)
	//!
	//! \param[in] cpu The CPU that triggered the call, if any
	//! \param[in] hint A hint about what kind of change triggered this call
	//! \param[in] numTasks If hint == ADDED_TASKS, numTasks is the amount
	//! of tasks added
	static inline void executeCPUManagerPolicy(
		ComputePlace *cpu,
		CPUManagerPolicyHint hint,
		size_t numTasks = 0
	) {
		assert(_cpuManager != nullptr);

		_cpuManager->executeCPUManagerPolicy(cpu, hint, numTasks);
	}

	//! \brief Get the maximum number of CPUs that will be used by the runtime
	//!
	//! \return The maximum number of CPUs that the runtime will ever use in
	//! the current execution
	static inline long getTotalCPUs()
	{
		assert(_cpuManager != nullptr);

		return _cpuManager->getTotalCPUs();
	}

	//! \brief Get the number of CPUs available through the process' mask
	//!
	//! \return The number of CPUs owned by the runtime
	static inline long getAvailableCPUs()
	{
		assert(_cpuManager != nullptr);

		return _cpuManager->getAvailableCPUs();
	}

	//! \brief Get a CPU object given a numerical system CPU identifier
	//!
	//! \param[in] systemCPUId The identifier
	//!
	//! \return The CPU object
	static inline CPU *getCPU(size_t systemCPUId)
	{
		assert(_cpuManager != nullptr);

		return _cpuManager->getCPU(systemCPUId);
	}

	//! \brief Try to obtain an unused CPU
	//!
	//! \return A CPU or nullptr
	static inline CPU *getUnusedCPU()
	{
		assert(_cpuManager != nullptr);

		return _cpuManager->getUnusedCPU();
	}

	//! \brief Get a reference to the list of CPUs
	//!
	//! \return A vector with all the CPU objects
	static inline std::vector<CPU *> const &getCPUListReference()
	{
		assert(_cpuManager != nullptr);

		return _cpuManager->getCPUListReference();
	}


	/*    CPUACTIVATION BRIDGE    */

	//! \brief Check and/or complete status transitions of a CPU onto which a
	//! thread is running
	//!
	//! \param[in,out] thread The thread which executes on the CPU we check for
	//!
	//! \return The current status of the CPU
	static inline CPU::activation_status_t checkCPUStatusTransitions(WorkerThread *thread)
	{
		assert(_cpuManager != nullptr);

		return _cpuManager->checkCPUStatusTransitions(thread);
	}

	//! \brief Check whether a CPU accepts work
	//!
	//! \param[in,out] cpu The CPU to check for
	//!
	//! \return Whether the CPU accepts work
	static inline bool acceptsWork(CPU *cpu)
	{
		assert(_cpuManager != nullptr);

		return _cpuManager->acceptsWork(cpu);
	}

	//! \brief Try to enable a CPU by its identifier
	//!
	//! \param[in] systemCPUId The identifier of the CPU to enable
	//!
	//! \return Whether the CPU was enabled
	static inline bool enable(size_t systemCPUId)
	{
		assert(_cpuManager != nullptr);

		return _cpuManager->enable(systemCPUId);
	}

	//! \brief Try to disable a CPU by its identifier
	//!
	//! \param[in] systemCPUId The identifier of the CPU to disable
	//!
	//! \return Whether the CPU was disabled
	static inline bool disable(size_t systemCPUId)
	{
		assert(_cpuManager != nullptr);

		return _cpuManager->disable(systemCPUId);
	}

	//! \brief Forcefully resume a CPU if it is paused
	//!
	//! \param[in] cpuId The id of the CPU to resume
	static inline void forcefullyResumeCPU(size_t cpuId)
	{
		assert(_cpuManager != nullptr);

		_cpuManager->forcefullyResumeCPU(cpuId);
	}


	/*    SHUTDOWN CPUS    */

	//! \brief Mark that a CPU is able to participate in the shutdown process
	//!
	//! \param[in] cpu The CPU object to offer
	static inline void addShutdownCPU(CPU *cpu)
	{
		assert(_cpuManager != nullptr);

		_cpuManager->addShutdownCPU(cpu);
	}

	//! \brief Try to obtain an unused CPU to participate in the shutdown
	//!
	//! \return A CPU or nullptr
	static inline CPU *getShutdownCPU()
	{
		assert(_cpuManager != nullptr);

		return _cpuManager->getShutdownCPU();
	}


	/*    TASKFORS    */

	//! \brief Get the number of taskfor groups
	//!
	//! \return The number of taskfor groups
	static inline size_t getNumTaskforGroups()
	{
		assert(_cpuManager != nullptr);

		return _cpuManager->getNumTaskforGroups();
	}

	//! \brief Get the number of CPUs that can collaborate to execute a single
	//! taskfor. I.e. the number of CPUs per taskfor group
	static inline size_t getNumCPUsPerTaskforGroup()
	{
		assert(_cpuManager != nullptr);

		return _cpuManager->getNumCPUsPerTaskforGroup();
	}

};


#endif // CPU_MANAGER_HPP
