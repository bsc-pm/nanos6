/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_MANAGER_INTERFACE_HPP
#define CPU_MANAGER_INTERFACE_HPP

#include <atomic>
#include <cassert>
#include <mutex>
#include <sched.h>
#include <sstream>
#include <vector>

#include <boost/dynamic_bitset.hpp>

#include "CPU.hpp"
#include "hardware/HardwareInfo.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "lowlevel/SpinLock.hpp"


enum CPUManagerPolicyHint {
	IDLE_CANDIDATE,
	ADDED_TASKS,
	HANDLE_TASKFOR
};


class CPUManagerInterface {

protected:
	
	//! Indicates if the thread manager has finished initializing the CPUs
	static std::atomic<bool> _finishedCPUInitialization;
	
	//! NUMA node CPU mask
	static std::vector<boost::dynamic_bitset<>> _NUMANodeMask;
	
	//! Map from system to virtual CPU id
	static std::vector<size_t> _systemToVirtualCPUId;
	
	//! Available CPUs indexed by virtual CPU identifier
	static std::vector<CPU *> _cpus;
	
	//! The process CPU mask
	static cpu_set_t _cpuMask;
	
	//! Spinlock to access idle CPUs
	static SpinLock _idleCPUsLock;
	
	//! The current number of idle CPUs, kept atomic through idleCPUsLock
	static size_t _numIdleCPUs;
	
	//! Idle CPUS
	static boost::dynamic_bitset<> _idleCPUs;
	
	//! Number of groups that can collaborate executing a single taskfor
	static EnvironmentVariable<size_t> _taskforGroups;
	
	
protected:
	
	//! \brief Instrument-related private function
	void reportInformation(size_t numSystemCPUs, size_t numNUMANodes);
	
	//! \brief Taskfor-related, get the closest number of taskfor groups
	//!
	//! \param[in] numCPUs The number of available CPUs
	//! \param[in] numGroups The number of groups specified by users
	inline size_t getClosestGroupNumber(size_t numCPUs, size_t numGroups) const
	{
		size_t result  = 0;
		size_t greater = numGroups + 1;
		size_t lower   = numGroups - 1;
		
		while (true) {
			if ((lower > 0) && (numCPUs % lower == 0)) {
				result = lower;
				break;
			}
			
			if ((greater <= numCPUs) && (numCPUs % greater == 0)) {
				result = greater;
				break;
			}
			
			lower--;
			greater++;
		}
		
		assert(result != 0);
		assert(result > 0 && result <= numCPUs);
		
		return result;
	}
	
	
public:
	
	/*    CPU MANAGER    */
	
	virtual inline ~CPUManagerInterface()
	{
	}
	
	//! \brief Pre-initialize structures for the CPUManager
	virtual void preinitialize();
	
	//! \brief Initialize all structures for the CPUManager
	virtual void initialize();
	
	//! \brief Notify all available CPUs that the runtime is shutting down
	virtual void shutdownPhase1() = 0;
	
	//! \brief Destroy any CPU-related structures
	virtual void shutdownPhase2() = 0;
	
	//! \brief Taking into account the current workload and the amount of
	//! active or idle CPUs, consider idling/waking up CPUs
	//!
	//! \param[in] cpu The CPU that triggered the call, if any
	//! \param[in] hint A hint about what kind of change triggered this call
	//! \param[in] numTasks A hint to be used by the policy taking actions,
	//! which contains information about what triggered a call to the policy
	virtual void executeCPUManagerPolicy(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numTasks = 0);
	
	//! \brief Check if CPU initialization has finished
	virtual inline bool hasFinishedInitialization() const
	{
		return _finishedCPUInitialization;
	}
	
	//! \brief Get a CPU object given a numerical system CPU identifier
	//!
	//! \param[in] systemCPUId The identifier
	//! \return The CPU object
	virtual inline CPU *getCPU(size_t systemCPUId) const
	{
		// _cpus is sorted by virtual ID
		assert(_systemToVirtualCPUId.size() > systemCPUId);
		
		size_t virtualCPUId = _systemToVirtualCPUId[systemCPUId];
		return _cpus[virtualCPUId];
	}
	
	//! \brief Get the maximum number of CPUs that will be used
	virtual inline long getTotalCPUs() const
	{
		return _cpus.size();
	}
	
	//! \brief Get a reference to the list of CPUs
	virtual inline std::vector<CPU *> const &getCPUListReference() const
	{
		return _cpus;
	}
	
	//! \brief Get the number of CPUs available through the process' mask
	virtual inline long getAvailableCPUs() const
	{
		return CPU_COUNT(&_cpuMask);
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
	//! \param[in,out] systemCPUId The identifier of the CPU to enable
	//!
	//! \return Whether the CPU was enabled
	virtual bool enable(size_t systemCPUId) = 0;
	
	//! \brief Try to disable a CPU by its identifier
	//!
	//! \param[in,out] systemCPUId The identifier of the CPU to disable
	//!
	//! \return Whether the CPU was disabled
	virtual bool disable(size_t systemCPUId) = 0;
	
	
	/*    IDLE CPUS    */
	
	//! \brief Mark a CPU as idle
	//!
	//! \param[in] cpu The CPU object to idle
	//! \param[in] inShutdown Whether the CPU becomes idle due to the runtime
	//! shutting down
	//!
	//! \return Whether the operation was successful
	virtual bool cpuBecomesIdle(CPU *cpu, bool inShutdown = false);
	
	//! \brief Get any idle CPU
	//!
	//! \param[in] inShutdown Whether the returned CPU is needed because the
	//! runtime is shutting down
	virtual CPU *getIdleCPU(bool = false);
	
	//! \brief Get a specific number of idle CPUs
	//!
	//! \param[in,out] idleCPUs A vector of at least size 'numCPUs' where the
	//! retreived idle CPUs will be placed
	//! \param[in] numCPUs The amount of CPUs to retreive
	//! \return The number of idle CPUs obtained/valid references in the vector
	virtual size_t getIdleCPUs(std::vector<CPU *> &idleCPUs, size_t numCPUs);
	
	//! \brief Get an idle CPU from a specific NUMA node
	virtual CPU *getIdleNUMANodeCPU(size_t NUMANodeId);
	
	//! \brief Mark a CPU as not idle (if possible)
	virtual bool unidleCPU(CPU *cpu);
	
	//! \brief Get all the idle CPUs that can collaborate in a taskfor
	//!
	//! \param[in,out] idleCPUs A vector where the unidled collaborators will
	//! be stored
	//! \param[in] cpu The CPU that created the taskfor
	virtual void getIdleCollaborators(std::vector<CPU *> &idleCPUs, ComputePlace *cpu);
	
	
	/*    TASKFORS    */
	
	//! \brief Get the number of taskfor groups
	virtual inline size_t getNumTaskforGroups() const
	{
		return _taskforGroups;
	}
	
	//! \brief Get the number of CPUs that can collaborate to execute a single
	//! taskfor. I.e. the number of CPUs per taskfor group
	virtual inline size_t getNumCPUsPerTaskforGroup() const
	{
		return HardwareInfo::getComputePlaceCount(nanos6_host_device) / _taskforGroups;
	}
	
};


#endif // CPU_MANAGER_INTERFACE_HPP
