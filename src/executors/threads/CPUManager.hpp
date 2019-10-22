/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_MANAGER_HPP
#define CPU_MANAGER_HPP


#include <mutex>
#include <atomic>

#include <boost/dynamic_bitset.hpp>

#include "CPU.hpp"
#include "CPUManager.hpp"
#include "CPUManagerPolicyInterface.hpp"
#include "ThreadManager.hpp"
#include "WorkerThread.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "lowlevel/SpinLock.hpp"

#include <InstrumentComputePlaceManagement.hpp>


class CPUManager {
private:
	//! \brief Available CPUs indexed by virtual CPU identifier
	static std::vector<CPU *> _cpus;
	
	//! \brief indicates if the thread manager has finished initializing the CPUs
	static std::atomic<bool> _finishedCPUInitialization;
	
	//! \brief threads blocked due to idleness
	static boost::dynamic_bitset<> _idleCPUs;
	
	//! \brief NUMA node CPU mask
	static std::vector<boost::dynamic_bitset<>> _NUMANodeMask;
	
	//! \brief Map from system to virtual CPU id
	static std::vector<size_t> _systemToVirtualCPUId;
	
	static SpinLock _idleCPUsLock;
	
	//! \brief Number of different groups that can collaborate to execute a single taskfor
	static EnvironmentVariable<size_t> _taskforGroups;
	
	//! The current number of idle CPUs, kept atomic through idleCPUsLock
	static size_t _numIdleCPUs;
	
	//! The policy used to manage CPUs
	static CPUManagerPolicyInterface *_cpuManagerPolicy;
	
	static void reportInformation(size_t numSystemCPUs, size_t numNUMANodes);
	
	static size_t getClosestGroupNumber(size_t numCPUs, size_t numGroups);
	
public:
	static void preinitialize();
	
	static void initialize();
	
	//! \brief Notify all available CPUs that the runtime is shutting down
	static void shutdownPhase1();
	
	//! \brief Complete the shutdown of the CPUManager
	static void shutdownPhase2();
	
	//! \brief get the CPU object assigned to a given numerical system CPU identifier
	static inline CPU *getCPU(size_t systemCPUId);
	
	//! \brief get the maximum number of CPUs that will be used
	static inline long getTotalCPUs();
	
	//! \brief check if initialization has finished
	static inline bool hasFinishedInitialization();
	
	//! \brief get a reference to the list of CPUs
	static inline std::vector<CPU *> const &getCPUListReference();
	
	//! \brief Mark a CPU as idle
	//!
	//! \param[in] inShutdown Whether the CPU is put idle so that any thread
	//! may use it for the shutdown process
	//!
	//! \return Whether the operation was successful
	static bool cpuBecomesIdle(CPU *cpu, bool inShutdown = false);
	
	//! \brief get an idle CPU
	static CPU *getIdleCPU();
	
	//! \brief Get a specific number of idle CPUs
	//!
	//! \param[in,out] idleCPUs A vector of at least size 'numCPUs' where the
	//! retreived idle CPUs will be placed
	//! \param[in] numCPUs The amount of CPUs to retreive
	//! \return The number of idle CPUs obtained/valid references in the vector
	static size_t getIdleCPUs(std::vector<CPU *> &idleCPUs, size_t numCPUs);
	
	//! \brief get an idle CPU from a specific NUMA node
	static CPU *getIdleNUMANodeCPU(size_t NUMANodeId);
	
	//! \brief mark a CPU as not being idle (if possible)
	static bool unidleCPU(CPU *cpu);
	
	//! \brief Get all the idle CPUs that can collaborate in a taskfor
	//!
	//! \param[in,out] idleCPUs A vector where the unidled collaborators will
	//! be stored
	//! \param[in] cpu The CPU that created the taskfor
	static void getIdleCollaborators(std::vector<CPU *> &idleCPUs, ComputePlace *cpu);
	
	//! \brief Get number of taskfor groups.
	static size_t getNumTaskforGroups();
	
	//! \brief Get number of CPUs that can collaborate to execute a single taskfor. i.e. number of CPUs per taskfor group.
	static size_t getNumCPUsPerTaskforGroup();
	
	//! \brief Taking into account the current workload and the amount of
	//! active or idle CPUs, consider idling/waking up CPUs
	//!
	//! \param[in] cpu The CPU that triggered the call, if any
	//! \param[in] hint A hint what kind of change triggered this call
	//! \param[in] numTasks A hint to be used by the policy taking actions,
	//! which contains information about what triggered a call to the policy
	static void executeCPUManagerPolicy(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numTasks = 0);
};


inline CPU *CPUManager::getCPU(size_t systemCPUId)
{
	// _cpus is sorted by virtual ID
	assert(_systemToVirtualCPUId.size() > systemCPUId);
	size_t virtualCPUId = _systemToVirtualCPUId[systemCPUId];
	
	return _cpus[virtualCPUId];
}

inline long CPUManager::getTotalCPUs()
{
	return _cpus.size();
}

inline bool CPUManager::hasFinishedInitialization()
{
	return _finishedCPUInitialization;
}


inline std::vector<CPU *> const &CPUManager::getCPUListReference()
{
	return _cpus;
}

inline size_t CPUManager::getNumTaskforGroups()
{
	return _taskforGroups;
}

inline size_t CPUManager::getClosestGroupNumber(size_t numCPUs, size_t numGroups)
{
	size_t result = 0;
	size_t greater = numGroups+1;
	size_t lower = numGroups-1;
	while (true) {
		if (lower > 0 && numCPUs % lower == 0) {
			result = lower;
			break;
		}
		if (greater <= numCPUs && numCPUs % greater == 0) {
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
#endif // CPU_MANAGER_HPP
