/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_MANAGER_INTERFACE_HPP
#define CPU_MANAGER_INTERFACE_HPP

#include <atomic>
#include <cassert>
#include <mutex>
#include <sched.h>
#include <vector>

#include <boost/dynamic_bitset.hpp>

#include "CPU.hpp"
#include "WorkerThread.hpp"
#include "hardware/HardwareInfo.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "lowlevel/SpinLock.hpp"


enum CPUManagerPolicyHint {
	IDLE_CANDIDATE,
	ADDED_TASKS,
	HANDLE_TASKFOR
};


class CPUManagerInterface {

protected:

	//! A vector of available CPUs
	static std::vector<CPU *> _cpus;

	//! The process' CPU mask
	static cpu_set_t _cpuMask;

	//! NUMA node CPU mask
	static std::vector<boost::dynamic_bitset<>> _NUMANodeMask;

	//! Map from system to virtual CPU id
	static std::vector<size_t> _systemToVirtualCPUId;

	//! Indicates the initialization of CPUs has finished
	static std::atomic<bool> _finishedCPUInitialization;

	//! The chosen number of taskfor groups
	static EnvironmentVariable<size_t> _taskforGroups;

	//! Whether we should emit a report with info about the taskfor groups.
	static EnvironmentVariable<bool> _taskforGroupsReportEnabled;


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
	inline void refineTaskforGroups(size_t numCPUs, size_t numNUMANodes)
	{
		// Whether the taskfor group envvar already has a value
		bool taskforGroupsSetByUser = _taskforGroups.isPresent();

		// Final warning message (only one)
		bool mustEmitWarning = false;
		std::ostringstream warningMessage;

		// The default value is the closest to 1 taskfor group per NUMA node
		if (!taskforGroupsSetByUser) {
			size_t closestGroups = numNUMANodes;
			if (numCPUs % numNUMANodes != 0) {
				closestGroups = getClosestGroupNumber(numCPUs, numNUMANodes);
				assert(numCPUs % closestGroups == 0);
			}
			_taskforGroups.setValue(closestGroups);
		} else {
			if (numCPUs < _taskforGroups) {
				warningMessage
					<< "More groups requested than available CPUs. "
					<< "Using " << numCPUs << " groups of 1 CPU each instead";

				_taskforGroups.setValue(numCPUs);
				mustEmitWarning = true;
			} else if (_taskforGroups == 0 || numCPUs % _taskforGroups != 0) {
				size_t closestGroups = getClosestGroupNumber(numCPUs, _taskforGroups);
				assert(numCPUs % closestGroups == 0);

				size_t cpusPerGroup = numCPUs / closestGroups;
				warningMessage
					<< _taskforGroups << " groups requested. "
					<< "The number of CPUs is not divisible by the number of groups. "
					<< "Using " << closestGroups << " groups of " << cpusPerGroup
					<< " CPUs each instead";

				_taskforGroups.setValue(closestGroups);
				mustEmitWarning = true;
			}
		}

		if (mustEmitWarning) {
			FatalErrorHandler::warnIf(true, warningMessage.str());
		}

		assert((_taskforGroups <= numCPUs) && (numCPUs % _taskforGroups == 0));
	}

public:

	virtual ~CPUManagerInterface()
	{
	}

	/*    CPU MANAGER    */

	//! \brief Pre-initialize structures for the CPUManager
	virtual void preinitialize() = 0;

	//! \brief Initialize all structures for the CPUManager
	virtual void initialize() = 0;

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
	//! - Adding tasks in the scheduler (hint = ADDED_TASKS)
	//! - Execution of a taskfor (hint = HANDLE_TASKFOR)
	//! - Running out of tasks to execute (hint = IDLE_CANDIDATE)
	//!
	//! \param[in,out] cpu The CPU that triggered the call, if any
	//! \param[in] hint A hint about what kind of change triggered this call
	//! \param[in] numTasks If hint == ADDED_TASKS, numTasks is the amount
	//! of tasks added
	virtual void executeCPUManagerPolicy(
		ComputePlace *cpu,
		CPUManagerPolicyHint hint,
		size_t numTasks = 0
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


	/*    CPUACTIVATION BRIDGE    */

	//! \brief Check the status transitions of a CPU onto which a thread is
	//! running
	//!
	//! \param[in,out] thread The thread which executes on the CPU we check for
	//!
	//! \return The current status of the CPU
	virtual CPU::activation_status_t checkCPUStatusTransitions(WorkerThread *thread) = 0;

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

	//! \brief Forcefully resume a CPU if it is idle
	//!
	//! \param[in] systemCPUId The identifier of the CPU to resume
	virtual void forcefullyResumeCPU(size_t systemCPUId) = 0;


	/*    SHUTDOWN CPUS    */

	//! \brief Get an unused CPU to participate in the shutdown process
	//!
	//! \return A CPU or nullptr
	virtual CPU *getShutdownCPU() = 0;

	//! \brief Mark that a CPU is able to participate in the shutdown process
	//!
	//! \param[in,out] cpu The CPU object to offer
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
		return HardwareInfo::getComputePlaceCount(nanos6_host_device) / _taskforGroups;
	}

	//! \brief Emits a brief report with information of the taskfor groups
	virtual void reportTaskforGroupsInfo(const size_t numTaskforGroups, const size_t numCPUsPerTaskforGroup) = 0;

};


#endif // CPU_MANAGER_INTERFACE_HPP
