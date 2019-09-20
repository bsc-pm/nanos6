/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_ACTIVATION_HPP
#define CPU_ACTIVATION_HPP


#include <cassert>

#include "CPU.hpp"
#include "scheduling/Scheduler.hpp"
#include "ThreadManager.hpp"
#include "CPUManager.hpp"

#include <InstrumentComputePlaceManagement.hpp>


class CPUActivation {
public:
	
	//! \brief Set a CPU online
	static inline bool enable(size_t systemCPUId)
	{
		CPU *cpu = CPUManager::getCPU(systemCPUId);
		assert(cpu != nullptr);
		
		cpu->initializeIfNeeded();
		
		bool successful = false;
		while (!successful) {
			CPU::activation_status_t currentStatus = cpu->getActivationStatus();
			switch (currentStatus) {
				case CPU::uninitialized_status:
					assert(false);
					return false;
				case CPU::enabled_status:
					// No change
					successful = true;
					break;
				case CPU::enabling_status:
					// No change
					successful = true;
					break;
				case CPU::disabling_status:
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::enabled_status);
					break;
				case CPU::disabled_status:
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::enabling_status);
					// Wake up a thread to allow the state change to progress
					if (successful) {
						ThreadManager::resumeIdle(cpu);
					}
					break;
				case CPU::shutting_down_status:
					// If the runtime is shutting down, no change and return
					return false;
			}
		}
		
		return true;
	}
	
	//! \brief Set a CPU offline
	static inline bool disable(size_t systemCPUId)
	{
		CPU *cpu = CPUManager::getCPU(systemCPUId);
		assert(cpu != nullptr);
		
		bool successful = false;
		while (!successful) {
			CPU::activation_status_t currentStatus = cpu->getActivationStatus();
			switch (currentStatus) {
				case CPU::uninitialized_status:
					assert(false);
					return false;
				case CPU::enabled_status:
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::disabling_status);
					break;
				case CPU::enabling_status:
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::disabled_status);
					break;
				case CPU::disabling_status:
					// No change
					successful = true;
					break;
				case CPU::disabled_status:
					// No change
					successful = true;
					break;
				case CPU::shutting_down_status:
					// If the runtime is shutting down, no change and return
					return false;
			}
		}
		
		return true;
	}
	
	//! \brief Check and handle CPU activation transitions
	//!
	//! NOTE: This code must be run regularly from within WorkerThreads
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
					assert(false);
					break;
				case CPU::enabled_status:
				case CPU::shutting_down_status:
					// No change and return immediately
					return currentStatus;
				case CPU::enabling_status:
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::enabled_status);
					if (successful) {
						currentStatus = CPU::enabled_status;
						Instrument::resumedComputePlace(cpu->getInstrumentationId());
					}
					break;
				case CPU::disabling_status:
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::disabled_status);
					if (successful) {
						successful = false; // Loop again, since things may have changed
						
						// There is no available hardware place, so this thread becomes idle
						ThreadManager::addIdler(currentThread);
						currentThread->switchTo(nullptr);
					}
					break;
				case CPU::disabled_status:
					// The CPU is disabled, the thread will become idle
					Instrument::suspendingComputePlace(cpu->getInstrumentationId());
					ThreadManager::addIdler(currentThread);
					currentThread->switchTo(nullptr);
					break;
			}
		}
		
		// Return the current status, whether there was a change
		return currentStatus;
	}
	
	//! \brief Notify a CPU that the runtime is shutting down
	static inline void shutdownCPU(CPU *cpu)
	{
		assert(cpu != nullptr);
		
		bool successful = false;
		while (!successful) {
			CPU::activation_status_t currentStatus = cpu->getActivationStatus();
			switch (currentStatus) {
				case CPU::uninitialized_status:
					// The CPU should not be uninitialized
					assert(false);
					break;
				case CPU::enabled_status:
				case CPU::disabling_status:
				case CPU::disabled_status:
					// If the CPU is enabled, disabling, or disabled
					// simply change try to change the status to shut down
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::shutting_down_status);
					break;
				case CPU::enabling_status:
					// If the CPU is being enabled, change to shut down and
					// notify that it has resumed (if the change is possible)
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::shutting_down_status);
					if (successful) {
						Instrument::resumedComputePlace(cpu->getInstrumentationId());
					}
					break;
				case CPU::shutting_down_status:
					// If the runtime is shutting down, no change and return
					return;
			}
		}
	}
	
	//! \brief Check if a CPU is accepting new work
	static inline bool acceptsWork(CPU *cpu)
	{
		assert(cpu != nullptr);
		
		CPU::activation_status_t currentStatus = cpu->getActivationStatus();
		switch (currentStatus) {
			case CPU::enabled_status:
			case CPU::enabling_status:
				return true;
				break;
			case CPU::uninitialized_status:
			case CPU::disabling_status:
			case CPU::disabled_status:
			case CPU::shutting_down_status:
				return false;
				break;
		}
		
		assert("Unhandled CPU activation status" == nullptr);
		return false;
	}
	
};


#endif // CPU_ACTIVATION_HPP
