/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
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
	
	//! \brief set a CPU online
	static inline void enable(size_t systemCPUId)
	{
		if (ThreadManager::mustExit()) {
			return;
		}
		
		CPU *cpu = CPUManager::getCPU(systemCPUId);
		assert(cpu != nullptr);
		
		cpu->initializeIfNeeded();
		bool successful = false;
		
		while (!successful) {
			CPU::activation_status_t currentStatus = cpu->_activationStatus;
			switch (currentStatus) {
				case CPU::uninitialized_status:
					assert(false);
					break;
				case CPU::starting_status:
					// Keep iterating until the CPU has actually been initialized
					break;
				case CPU::enabled_status:
					// No change
					successful = true;
					break;
				case CPU::enabling_status:
					// No change
					successful = true;
					break;
				case CPU::disabling_status:
					successful = cpu->_activationStatus.compare_exchange_strong(currentStatus, CPU::enabled_status);
					break;
				case CPU::disabled_status:
					successful = cpu->_activationStatus.compare_exchange_strong(currentStatus, CPU::enabling_status);
					// Wake up a thread to allow the state change to progress
					if (successful) {
						ThreadManager::resumeIdle(cpu);
					}
					break;
			}
		}
	}
	
	//! \brief set a CPU offline
	static inline void disable(size_t systemCPUId)
	{
		if (ThreadManager::mustExit()) {
			return;
		}
		
		CPU *cpu = CPUManager::getCPU(systemCPUId);
		assert(cpu != nullptr);
		
		bool successful = false;
		
		while (!successful) {
			CPU::activation_status_t currentStatus = cpu->_activationStatus;
			switch (currentStatus) {
				case CPU::uninitialized_status:
					assert(false);
					break;
				case CPU::starting_status:
					// Keep iterating until the CPU has actually been initialized
					break;
				case CPU::enabled_status:
					successful = cpu->_activationStatus.compare_exchange_strong(currentStatus, CPU::disabling_status);
					break;
				case CPU::enabling_status:
					successful = cpu->_activationStatus.compare_exchange_strong(currentStatus, CPU::disabled_status);
					break;
				case CPU::disabling_status:
					// No change
					successful = true;
					break;
				case CPU::disabled_status:
					// No change
					successful = true;
					break;
			}
		}
	}
	
	//! \brief check and handle CPU activation transitions
	//!
	//! This code must be run regularly from within the worker threads
	static inline void activationCheck(WorkerThread *currentThread)
	{
		assert(currentThread != nullptr);
		
		bool successful = false;
		while (!successful) {
			CPU *cpu = currentThread->getComputePlace();
			assert(cpu != nullptr);
			
			CPU::activation_status_t currentStatus = cpu->_activationStatus;
			switch (currentStatus) {
				case CPU::uninitialized_status:
					assert(false);
					break;
				case CPU::starting_status:
					{
						__attribute__((unused)) bool enabled = cpu->_activationStatus.compare_exchange_strong(currentStatus, CPU::enabled_status);
						assert(enabled); // There should be no other thread changing this
						Instrument::resumedComputePlace(cpu->getInstrumentationId());
					}
					break;
				case CPU::enabled_status:
					// No change
					successful = true;
					break;
				case CPU::enabling_status:
					successful = cpu->_activationStatus.compare_exchange_strong(currentStatus, CPU::enabled_status);
					if (successful) {
						Instrument::resumedComputePlace(cpu->getInstrumentationId());
						Scheduler::enableComputePlace(cpu);
					}
					break;
				case CPU::disabling_status:
					successful = cpu->_activationStatus.compare_exchange_strong(currentStatus, CPU::disabled_status);
					if (successful) {
						// Mark the Hardware Place as disabled
						Scheduler::disableComputePlace(cpu);
						
						successful = false; // Loop again, since things may have changed
						
						ComputePlace *idleComputePlace = Scheduler::getIdleComputePlace();
						if (idleComputePlace != nullptr) {
							// Migrate the thread to the idle hardware place
							currentThread->migrate((CPU *) idleComputePlace);
						} else {
							// There is no available hardware place, so this thread becomes idle
							ThreadManager::addIdler(currentThread);
							currentThread->switchTo(nullptr);
						}
					}
					break;
				case CPU::disabled_status:
					if (!currentThread->hasPendingShutdown()) {
						Instrument::suspendingComputePlace(cpu->getInstrumentationId());
						ThreadManager::addIdler(currentThread);
						currentThread->switchTo(nullptr);
					} else {
						successful = true;
					}
					break;
			}
		}
	}
	
	//! \brief check if a CPU is accepting new work
	static inline bool acceptsWork(CPU *cpu)
	{
		assert(cpu != nullptr);
		
		CPU::activation_status_t currentStatus = cpu->_activationStatus;
		switch (currentStatus) {
			case CPU::starting_status:
			case CPU::enabled_status:
			case CPU::enabling_status:
				return true;
				break;
			case CPU::uninitialized_status:
			case CPU::disabling_status:
			case CPU::disabled_status:
				return false;
				break;
		}
		
		assert("Unhandled CPU activation status" == nullptr);
		return false;
	}
	
	//! \brief check if the CPU has actually been started
	static inline bool hasStarted(CPU *cpu)
	{
		assert(cpu != nullptr);
		
		return (cpu->_activationStatus != CPU::starting_status);
	}

	//! \brief check if the CPU is being initialized 
	static inline bool isBeingInitialized(CPU *cpu) 
	{
		assert(cpu != nullptr);
		
		return (cpu->_activationStatus == CPU::starting_status);
	}
	
};


#endif // CPU_ACTIVATION_HPP
