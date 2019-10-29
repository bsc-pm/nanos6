/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
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
	}
	
	//! \brief Taking into account the current workload and the amount of
	//! active or idle CPUs, consider idling/waking up CPUs
	//!
	//! \param[in] cpu The CPU that triggered the call, if any
	//! \param[in] hint A hint about what kind of change triggered this call
	//! \param[in] numTasks A hint to be used by the policy taking actions,
	//! which contains information about what triggered a call to the policy
	static inline void executeCPUManagerPolicy(ComputePlace *cpu, CPUManagerPolicyHint hint, size_t numTasks = 0)
	{
		assert(_cpuManager != nullptr);
		
		_cpuManager->executeCPUManagerPolicy(cpu, hint, numTasks);
	}
	
	//! \brief Check if CPU initialization has finished
	static inline bool hasFinishedInitialization()
	{
		assert(_cpuManager != nullptr);
		
		return _cpuManager->hasFinishedInitialization();
	}
	
	//! \brief Get a CPU object given a numerical system CPU identifier
	//!
	//! \param[in] systemCPUId The identifier
	//! \return The CPU object
	static inline CPU *getCPU(size_t systemCPUId)
	{
		assert(_cpuManager != nullptr);
		
		return _cpuManager->getCPU(systemCPUId);
	}
	
	//! \brief Get the maximum number of CPUs that will be used
	static inline long getTotalCPUs()
	{
		assert(_cpuManager != nullptr);
		
		return _cpuManager->getTotalCPUs();
	}
	
	//! \brief Get the number of CPUs available through the process' mask
	static inline long getAvailableCPUs()
	{
		assert(_cpuManager != nullptr);
		
		return _cpuManager->getAvailableCPUs();
	}
	
	//! \brief Get a reference to the list of CPUs
	static inline std::vector<CPU *> const &getCPUListReference()
	{
		assert(_cpuManager != nullptr);
		
		return _cpuManager->getCPUListReference();
	}
	
	
	/*    CPUACTIVATION BRIDGE    */
	
	//! \brief Check the status transitions of a CPU onto which a thread is
	//! running
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
	//! \param[in,out] systemCPUId The identifier of the CPU to enable
	//!
	//! \return Whether the CPU was enabled
	static inline bool enable(size_t systemCPUId)
	{
		assert(_cpuManager != nullptr);
		
		return _cpuManager->enable(systemCPUId);
	}
	
	//! \brief Try to disable a CPU by its identifier
	//!
	//! \param[in,out] systemCPUId The identifier of the CPU to disable
	//!
	//! \return Whether the CPU was disabled
	static inline bool disable(size_t systemCPUId)
	{
		assert(_cpuManager != nullptr);
		
		return _cpuManager->disable(systemCPUId);
	}
	
	
	/*    IDLE CPUS    */
	
	//! \brief Mark a CPU as idle
	//!
	//! \param[in] cpu The CPU object to idle
	//! \param[in] inShutdown Whether the CPU becomes idle due to the runtime
	//! shutting down
	//!
	//! \return Whether the operation was successful
	static inline bool cpuBecomesIdle(CPU *cpu, bool inShutdown = false)
	{
		assert(_cpuManager != nullptr);
		
		return _cpuManager->cpuBecomesIdle(cpu, inShutdown);
	}
	
	//! \brief Get any idle CPU
	//!
	//! \param[in] inShutdown Whether the returned CPU is needed because the
	//! runtime is shutting down
	static inline CPU *getIdleCPU(bool inShutdown = false)
	{
		assert(_cpuManager != nullptr);
		
		return _cpuManager->getIdleCPU(inShutdown);
	}
	
	//! \brief Get a specific number of idle CPUs
	//!
	//! \param[in,out] idleCPUs A vector of at least size 'numCPUs' where the
	//! retreived idle CPUs will be placed
	//! \param[in] numCPUs The amount of CPUs to retreive
	//! \return The number of idle CPUs obtained/valid references in the vector
	static inline size_t getIdleCPUs(std::vector<CPU *> &idleCPUs, size_t numCPUs)
	{
		assert(_cpuManager != nullptr);
		
		return _cpuManager->getIdleCPUs(idleCPUs, numCPUs);
	}
	
	//! \brief Get an idle CPU from a specific NUMA node
	static inline CPU *getIdleNUMANodeCPU(size_t NUMANodeId)
	{
		assert(_cpuManager != nullptr);
		
		return _cpuManager->getIdleNUMANodeCPU(NUMANodeId);
	}
	
	//! \brief Mark a CPU as not idle (if possible)
	static inline bool unidleCPU(CPU *cpu)
	{
		assert(_cpuManager != nullptr);
		
		return _cpuManager->unidleCPU(cpu);
	}
	
	//! \brief Get all the idle CPUs that can collaborate in a taskfor
	//!
	//! \param[in,out] idleCPUs A vector where the unidled collaborators will
	//! be stored
	//! \param[in] cpu The CPU that created the taskfor
	static inline void getIdleCollaborators(std::vector<CPU *> &idleCPUs, ComputePlace *cpu)
	{
		assert(_cpuManager != nullptr);
		
		_cpuManager->getIdleCollaborators(idleCPUs, cpu);
	}
	
	
	/*    TASKFORS    */
	
	//! \brief Get the number of taskfor groups
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
