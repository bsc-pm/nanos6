/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef DLB_CPU_ACTIVATION_HPP
#define DLB_CPU_ACTIVATION_HPP

#include <cassert>
#include <ctime>
#include <dlb.h>

#include "executors/threads/CPU.hpp"
#include "executors/threads/CPUManager.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "scheduling/Scheduler.hpp"

#include <InstrumentComputePlaceManagement.hpp>
#include <Monitoring.hpp>


class CPUActivation {

private:
	
	//! Amount of nanoseconds to delay if a CPU wants to be enabled but is
	//! being used by another process
	static timespec _delayCPUEnabling;
	
	
private:
	
	//! \brief Lend a CPU to another runtime or process
	//!
	//! \param[in] systemCPUId The id of the CPU to be lent
	static inline void dlbLendCPU(size_t systemCPUId)
	{
		int ret = DLB_LendCpu(systemCPUId);
		if (ret != DLB_SUCCESS) {
			FatalErrorHandler::failIf(
				true,
				"DLB Error ", ret, " when trying to lend a CPU(",
				systemCPUId, ")"
			);
		}
	}
	
	//! \brief Reclaim a previously lent CPU
	//!
	//! \param[in] systemCPUId The id of the CPU to be reclaimed
	static inline void dlbReclaimCPU(size_t systemCPUId)
	{
		int ret = DLB_ReclaimCpu(systemCPUId);
		if (ret != DLB_NOUPDT && ret != DLB_NOTED && ret != DLB_SUCCESS) {
			FatalErrorHandler::failIf(
				true,
				"DLB Error ", ret, " when trying to reclaim a CPU(",
				systemCPUId, ")"
			);
		}
	}
	
	//! \brief Reclaim previously lent CPUs
	//!
	//! \param[in] numCPUs The number of CPUs to reclaim
	static inline void dlbReclaimCPUs(size_t numCPUs)
	{
		int ret = DLB_ReclaimCpus(numCPUs);
		if (ret != DLB_NOUPDT && ret != DLB_NOTED && ret != DLB_SUCCESS) {
			FatalErrorHandler::failIf(
				true,
				"DLB Error ", ret, " when trying to reclaim ", numCPUs, " CPUs"
			);
		}
	}
	
	//! \brief Reclaim previously lent CPUs
	//!
	//! \param[in] maskCPUs A mask signaling which CPUs to reclaim
	static inline void dlbReclaimCPUs(const cpu_set_t &maskCPUs)
	{
		int ret = DLB_ReclaimCpuMask(&(maskCPUs));
		if (ret != DLB_NOUPDT && ret != DLB_NOTED && ret != DLB_SUCCESS) {
			FatalErrorHandler::failIf(
				true,
				"DLB Error ", ret, " when reclaiming CPUs through a mask"
			);
		}
	}
	
	
public:
	
	//! \brief Check if a CPU is accepting new work
	static inline bool acceptsWork(CPU *cpu)
	{
		assert(cpu != nullptr);
		
		CPU::activation_status_t currentStatus = cpu->getActivationStatus();
		switch (currentStatus) {
			case CPU::enabled_status:
			case CPU::enabling_status:
				return true;
			case CPU::uninitialized_status:
			case CPU::lent_status:
			case CPU::lending_status:
			case CPU::shutdown_status:
			case CPU::shutting_down_status:
				return false;
			case CPU::disabled_status:
			case CPU::disabling_status:
				assert(false);
				return false;
		}
		
		assert("Unhandled CPU activation status" == nullptr);
		return false;
	}
	
	//! \brief Enable a CPU
	//! NOTE: The enable/disable API has no effect if DLB is enabled
	//! (i.e., it has no effect in this implementation)
	//!
	//! \return False in this implementation
	static inline bool enable(size_t)
	{
		FatalErrorHandler::warnIf(
			true,
			"The enable/disable API is deactivated when using DLB. ",
			"No action will be performed"
		);
		
		return false;
	}
	
	//! \brief Disable a CPU
	//! NOTE: The enable/disable API has no effect if DLB is enabled
	//! (i.e., it has no effect in this implementation)
	//!
	//! \return False in this implementation
	static inline bool disable(size_t)
	{
		FatalErrorHandler::warnIf(
			true,
			"The enable/disable API is deactivated when using DLB. ",
			"No action will be performed"
		);
		
		return false;
	}
	
	//! \brief Lend a CPU to another runtime
	//!
	//! \param[in] cpu The CPU to lend
	static inline void lendCPU(CPU *cpu)
	{
		assert(cpu != nullptr);
		
		// NOTE: The CPU should only be lent when no work is available from the
		// CPUManager's policy. Thus, we take for granted that the status of
		// the CPU is 'enabled_status' (unless it is shutting down)
		CPU::activation_status_t expectedStatus = CPU::enabled_status;
		bool successful = cpu->getActivationStatus().compare_exchange_strong(expectedStatus, CPU::lending_status);
		if (successful) {
			WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
			assert(currentThread != nullptr);
			
			// Lend the CPU
			dlbLendCPU(cpu->getSystemCPUId());
			
			expectedStatus = CPU::lending_status;
			successful = cpu->getActivationStatus().compare_exchange_strong(expectedStatus, CPU::lent_status);
			assert(successful);
			
			// The thread becomes idle
			ThreadManager::addIdler(currentThread);
			currentThread->switchTo(nullptr);
			
			// NOTE: When a thread is resumed in the CPU we've just lent, the
			// CPU may not be available yet. We check this through DLB in
			// 'checkCPUStatusTransitions'
		} else {
			// The status change should always work, as noted previously
			assert(cpu->getActivationStatus() == CPU::shutdown_status);
		}
	}
	
	//! \brief Try to reclaim a specific number of CPUs
	//!
	//! \param[in] numCPUs The number of CPUs to reclaim
	static inline void reclaimCPUs(size_t numCPUs)
	{
		assert(numCPUs > 0);
		
		dlbReclaimCPUs(numCPUs);
	}
	
	//! \brief Try to reclaim a specific number of CPUs
	//!
	//! \param[in] maskCPUs A mask signaling which CPUs to reclaim
	static inline void reclaimCPUs(const cpu_set_t &maskCPUs)
	{
		assert(CPU_COUNT(&maskCPUs) > 0);
		
		dlbReclaimCPUs(maskCPUs);
	}
	
	//! \brief DLB-specific callback called when a CPU is returned to us
	//!
	//! \param[in] systemCPUId The id of the CPU that can be used once again
	//! \param[in] args DLB-compliant callback arguments
	static inline void dlbEnableCallback(int systemCPUId, void *)
	{
		CPU *cpu = CPUManager::getCPU(systemCPUId);
		assert(cpu != nullptr);
		
		bool successful = false;
		while (!successful) {
			CPU::activation_status_t currentStatus = cpu->getActivationStatus();
			switch (currentStatus) {
				case CPU::uninitialized_status:
				case CPU::disabled_status:
				case CPU::disabling_status:
					// The CPU should never be in here in this implementation
					assert(false);
					return;
				case CPU::enabled_status:
				case CPU::enabling_status:
				case CPU::shutdown_status:
					// If we are not expecting a callback, ignore it
					successful = true;
					break;
				case CPU::lending_status:
					// In this case we iterate until the lending is complete
					break;
				case CPU::lent_status:
					// If we are expecting a callback, try to reenable the CPU
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::enabling_status);
					if (successful) {
						// Resume a thread so it sees the status changes
						ThreadManager::resumeIdle(cpu);
					}
					break;
				case CPU::shutting_down_status:
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::shutdown_status);
					if (successful) {
						// Resume a thread so it sees the status change
						ThreadManager::resumeIdle(cpu, true, true);
					}
					break;
			}
		}
	}
	
	//! \brief Check and handle CPU activation transitions
	//! NOTE: This code must be run regularly from within WorkerThreads
	//!
	//! \param[in] currentThread The WorkerThread is currently checking
	//! transitions of the CPU it is running on
	//!
	//! \return The current status of the CPU the thread is running on
	static inline CPU::activation_status_t checkCPUStatusTransitions(WorkerThread *currentThread)
	{
		assert(currentThread != nullptr);
		CPU::activation_status_t currentStatus;
		
		bool successful = false;
		while (!successful) {
			CPU *cpu = currentThread->getComputePlace();
			assert(cpu != nullptr);
			
			currentStatus = cpu->getActivationStatus();
			switch (currentStatus) {
				case CPU::uninitialized_status:
				case CPU::disabled_status:
				case CPU::disabling_status:
				case CPU::lending_status:
					// The CPU should never be disabled in this implementation
					// nor see itself as lending
					assert(false);
					return currentStatus;
				case CPU::enabled_status:
				case CPU::shutdown_status:
					// If the CPU is enabled or shutdown, do nothing
					successful = true;
					break;
				case CPU::lent_status:
				case CPU::shutting_down_status:
					// If a thread is woken up in this CPU but it is not
					// supposed to be running, readd the thread as idle
					ThreadManager::addIdler(currentThread);
					currentThread->switchTo(nullptr);
					break;
				case CPU::enabling_status:
					// If the CPU is enabled, DLB may have woken us even though
					// another process may be using our CPU still
					// Before completing the enable, check if this is the case
					while (DLB_CheckCpuAvailability(cpu->getSystemCPUId()) != DLB_SUCCESS) {
						// The CPU is not ready yet, sleep for a bit
						nanosleep(&_delayCPUEnabling, nullptr);
					}
					
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::enabled_status);
					if (successful) {
						currentStatus = CPU::enabled_status;
						Instrument::resumedComputePlace(cpu->getInstrumentationId());
						Monitoring::cpuBecomesActive(cpu->getIndex());
					}
					break;
			}
		}
		
		// Return the current status, whether there was a change
		return currentStatus;
	}
	
	//! \brief Notify to a CPU that the runtime is shutting down
	//!
	//! \param[in] cpu The CPU that transitions to a shutdown status
	static inline void shutdownCPU(CPU *cpu)
	{
		assert(cpu != nullptr);
		
		bool successful = false;
		while (!successful) {
			CPU::activation_status_t currentStatus = cpu->getActivationStatus();
			switch (currentStatus) {
				case CPU::uninitialized_status:
				case CPU::disabled_status:
				case CPU::disabling_status:
				case CPU::shutdown_status:
				case CPU::shutting_down_status:
					// The CPU should never be disabled in this implementation
					assert(false);
					return;
				case CPU::enabled_status:
				case CPU::enabling_status:
					// If the CPU is enabled, change to shutdown
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::shutdown_status);
					break;
				case CPU::lent_status:
					// If the CPU is lent, transition to a shutting down status
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::shutting_down_status);
					if (successful) {
						dlbReclaimCPU(cpu->getSystemCPUId());
					}
					break;
				case CPU::lending_status:
					// Iterate until the CPU is lent
					break;
			}
		}
	}
	
};


#endif // DLB_CPU_ACTIVATION_HPP
