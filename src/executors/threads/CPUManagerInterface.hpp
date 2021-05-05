/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_MANAGER_INTERFACE_HPP
#define CPU_MANAGER_INTERFACE_HPP

#include <atomic>
#include <cassert>
#include <mutex>
#include <sched.h>
#include <string>
#include <vector>

#include <boost/dynamic_bitset.hpp>

#include "CPU.hpp"
#include "CPUManagerPolicyInterface.hpp"
#include "WorkerThread.hpp"
#include "hardware/HardwareInfo.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "lowlevel/SpinLock.hpp"
#include "support/config/ConfigVariable.hpp"


class CPUManagerInterface {

protected:

	//! A vector of available CPUs
	static std::vector<CPU *> _cpus;

	//! Map from system to virtual CPU id
	static std::vector<size_t> _systemToVirtualCPUId;

	//! The process' CPU mask
	static cpu_set_t _cpuMask;

	//! Indicates the initialization of CPUs has finished
	static std::atomic<bool> _finishedCPUInitialization;

	//! The chosen number of taskfor groups
	static ConfigVariable<size_t> _taskforGroups;

	//! Whether we should emit a report with info about the taskfor groups.
	static ConfigVariable<bool> _taskforGroupsReportEnabled;

	//! The decision-taking policy of the CPU Manager
	static CPUManagerPolicyInterface *_cpuManagerPolicy;

	//! The chosen CPU Manager policy
	static ConfigVariable<std::string> _policyChosen;

	//! The identifier of the policy
	static cpumanager_policy_t _policyId;

	//! The virtual id of the first owned CPU of this process
	static size_t _firstCPUId;

	//! The virtual CPU of the leader thread
	static CPU *_leaderThreadCPU;

private:

	//! \brief Taskfor-related, get the closest number of taskfor groups
	//!
	//! \param[in] numCPUs The number of available CPUs
	//! \param[in] numGroups The number of groups specified by users
	//!
	//! \return The closest number of taskfor groups to numGroups
	inline size_t getClosestGroupNumber(size_t numCPUs, size_t numGroups) const
	{
		size_t result = 0;

		// If the chosen value is impossible, get the closest maximum value
		if (numGroups == 0) {
			// 1 group of numCPUs CPUs
			return 1;
		} else if (numGroups > numCPUs) {
			// numCPUs groups of 1 CPU
			return numCPUs;
		}

		// The chosen value is somewhat decent, get its closest valid number
		size_t lower = numGroups - 1;
		size_t upper = numGroups + 1;
		while (lower > 0 || upper <= numCPUs) {
			if ((lower > 0) && (numCPUs % lower == 0)) {
				result = lower;
				break;
			}

			if ((upper <= numCPUs) && (numCPUs % upper == 0)) {
				result = upper;
				break;
			}

			// We should never underflow as we are working with size_t
			if (lower > 0) {
				lower--;
			}
			upper++;
		}

		assert((result > 0) && (result <= numCPUs) && (numCPUs % result == 0));

		return result;
	}

protected:

	//! \brief Instrument-related private function
	void reportInformation(size_t numSystemCPUs, size_t numNUMANodes);

	//! \brief Find the appropriate value for the taskfor groups env var
	//!
	//! \param[in] numCPUs The number of CPUs used by the runtime
	//! \param[in] numNUMANodes The number of NUMA nodes in the system
	void refineTaskforGroups(size_t numCPUs, size_t numNUMANodes);

	//! \brief Emits a brief report with information of the taskfor groups
	void reportTaskforGroupsInfo();

public:

	virtual ~CPUManagerInterface()
	{
	}

	/*    CPU MANAGER    */

	//! \brief Pre-initialize structures for the CPUManager
	virtual void preinitialize() = 0;

	//! \brief Initialize all structures for the CPUManager
	virtual void initialize() = 0;

	//! \brief Get the CPU manager's policy id
	inline cpumanager_policy_t getPolicyId()
	{
		return _policyId;
	}

	//! \brief Check whether DLB is enabled
	inline virtual bool isDLBEnabled() const
	{
		return false;
	}

	//! \brief Check if CPU initialization has finished
	inline bool hasFinishedInitialization() const
	{
		return _finishedCPUInitialization;
	}

	//! \brief Notify all available CPUs that the runtime is shutting down
	virtual void shutdownPhase1() = 0;

	//! \brief Destroy any CPU-related structures
	virtual void shutdownPhase2() = 0;

	//! \brief This method is executed after the amount of work in the runtime
	//! changes. Some common scenarios include:
	//! - Requesting the resume of idle CPUs (hint = REQUEST_CPUS)
	//! - Execution of a taskfor (hint = HANDLE_TASKFOR)
	//! - Running out of tasks to execute (hint = IDLE_CANDIDATE)
	//!
	//! \param[in] cpu The CPU that triggered the call, if any
	//! \param[in] hint A hint about what kind of change triggered this call
	//! \param[in] numRequested If hint == REQUEST_CPUS, numRequested is the amount
	//! of idle CPUs to resume
	virtual void executeCPUManagerPolicy(
		ComputePlace *cpu,
		CPUManagerPolicyHint hint,
		size_t numRequested = 0
	) = 0;

	//! \brief Get the maximum number of CPUs that will be used by the runtime
	//!
	//! \return The maximum number of CPUs that the runtime will ever use in
	//! the current execution
	inline long getTotalCPUs() const
	{
		return _cpus.size();
	}

	//! \brief Get the number of CPUs available through the process' mask
	inline long getAvailableCPUs() const
	{
		return CPU_COUNT(&_cpuMask);
	}

	//! \brief Get a CPU object given a numerical system CPU identifier
	//!
	//! \param[in] systemCPUId The identifier
	//!
	//! \return The CPU object
	virtual CPU *getCPU(size_t systemCPUId) const = 0;

	//! \brief Try to obtain an unused CPU
	//!
	//! \return A CPU or nullptr
	virtual CPU *getUnusedCPU() = 0;

	//! \brief Get a reference to the list of CPUs
	inline std::vector<CPU *> const &getCPUListReference() const
	{
		return _cpus;
	}

	//! \brief Forcefully resume the first CPU if it is idle
	virtual void forcefullyResumeFirstCPU() = 0;

	//! \brief Check whether a CPU is the first CPU of this process' mask
	//!
	//! \param[in] index The (virtual) index of the compute place
	inline bool isFirstCPU(size_t index) const
	{
		return (index == _firstCPUId);
	}

	//! \brief Get the virtual CPU of the leader thread
	inline CPU *getLeaderThreadCPU() const
	{
		return _leaderThreadCPU;
	}


	/*    CPUACTIVATION BRIDGE    */

	//! \brief Check the status transitions of a CPU onto which a thread is
	//! running
	//!
	//! \param[in,out] thread The thread which executes on the CPU we check for
	//!
	//! \return The current status of the CPU
	virtual CPU::activation_status_t checkCPUStatusTransitions(WorkerThread *thread) = 0;

	//! \brief Check whether the CPU has to be returned
	//!
	//! \param[in] thread The thread which executes on the CPU we check for
	virtual void checkIfMustReturnCPU(WorkerThread *thread) = 0;

	//! \brief Check whether a CPU accepts work
	//!
	//! \param[in,out] cpu The CPU to check for
	//!
	//! \return Whether the CPU accepts work
	virtual bool acceptsWork(CPU *cpu) = 0;

	//! \brief Try to enable a CPU by its identifier
	//!
	//! \param[in] systemCPUId The identifier of the CPU to enable
	//!
	//! \return Whether the CPU was enabled
	virtual bool enable(size_t systemCPUId) = 0;

	//! \brief Try to disable a CPU by its identifier
	//!
	//! \param[in] systemCPUId The identifier of the CPU to disable
	//!
	//! \return Whether the CPU was disabled
	virtual bool disable(size_t systemCPUId) = 0;


	/*    SHUTDOWN CPUS    */

	//! \brief Get an unused CPU to participate in the shutdown process
	//!
	//! \return A CPU or nullptr
	virtual CPU *getShutdownCPU() = 0;

	//! \brief Mark that a CPU is able to participate in the shutdown process
	//!
	//! \param[in] cpu The CPU object to offer
	virtual void addShutdownCPU(CPU *cpu) = 0;


	/*    TASKFORS    */

	//! \brief Get the number of taskfor groups
	//!
	//! \return The number of taskfor groups
	inline size_t getNumTaskforGroups() const
	{
		return _taskforGroups;
	}

	//! \brief Get the number of CPUs that can collaborate to execute a single
	//! taskfor. I.e. the number of CPUs per taskfor group
	inline size_t getNumCPUsPerTaskforGroup() const
	{
		return _cpus.size() / _taskforGroups;
	}

};


#endif // CPU_MANAGER_INTERFACE_HPP
