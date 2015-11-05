#ifndef CPU_ACTIVATION_HPP
#define CPU_ACTIVATION_HPP


#include <cassert>

#include "CPU.hpp"
#include "scheduling/Scheduler.hpp"
#include "ThreadManager.hpp"


class CPUActivation {
public:
	
	//! \brief check and handle CPU activation initialization
	//!
	//! This code must be run from within a worker thread within its initialization
	static inline void threadInitialization(WorkerThread *currentThread)
	{
		assert(currentThread != nullptr);
		
		CPU *cpu = currentThread->getHardwarePlace();
		assert(cpu != nullptr);
		
		CPU::activation_status_t currentStatus = cpu->_activationStatus;
		if (currentStatus == CPU::starting_status) {
			#if NDEBUG
				cpu->_activationStatus.compare_exchange_strong(currentStatus, CPU::enabled_status);
			#else
				bool successful = cpu->_activationStatus.compare_exchange_strong(currentStatus, CPU::enabled_status);
				assert(successful); // There should be no other thread changing this
			#endif
		}
	}
	
	
	//! \brief set a CPU online
	static inline void enable(size_t systemCPUId)
	{
		if (ThreadManager::mustExit()) {
			return;
		}
		
		CPU *cpu = ThreadManager::getCPU(systemCPUId);
		assert(cpu != nullptr);
		
		bool successful = false;
		
		while (!successful) {
			CPU::activation_status_t currentStatus = cpu->_activationStatus;
			switch (currentStatus) {
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
					successful = cpu->_activationStatus.compare_exchange_weak(currentStatus, CPU::enabled_status);
					break;
				case CPU::disabled_status:
					successful = cpu->_activationStatus.compare_exchange_weak(currentStatus, CPU::enabling_status);
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
		
		CPU *cpu = ThreadManager::getCPU(systemCPUId);
		assert(cpu != nullptr);
		
		bool successful = false;
		
		while (!successful) {
			CPU::activation_status_t currentStatus = cpu->_activationStatus;
			switch (currentStatus) {
				case CPU::starting_status:
					// Keep iterating until the CPU has actually been initialized
					break;
				case CPU::enabled_status:
					successful = cpu->_activationStatus.compare_exchange_weak(currentStatus, CPU::disabling_status);
					break;
				case CPU::enabling_status:
					successful = cpu->_activationStatus.compare_exchange_weak(currentStatus, CPU::disabled_status);
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
			CPU *cpu = currentThread->getHardwarePlace();
			assert(cpu != nullptr);
			
			CPU::activation_status_t currentStatus = cpu->_activationStatus;
			switch (currentStatus) {
				case CPU::starting_status:
					assert(false && "Invalid CPU activation status");
					break;
				case CPU::enabled_status:
					// No change
					successful = true;
					break;
				case CPU::enabling_status:
					successful = cpu->_activationStatus.compare_exchange_weak(currentStatus, CPU::enabled_status);
					break;
				case CPU::disabling_status:
					successful = cpu->_activationStatus.compare_exchange_weak(currentStatus, CPU::disabled_status);
					if (successful) {
						HardwarePlace *idleHardwarePlace = Scheduler::getIdleHardwarePlace();
						assert(idleHardwarePlace != cpu);
						
						if (idleHardwarePlace != nullptr) {
							ThreadManager::resumeIdle((CPU *) idleHardwarePlace);
						}
						
						ThreadManager::addIdler(currentThread);
						ThreadManager::switchThreads(currentThread, nullptr);
						successful = false; // Loop again, since the CPU has been or is being re-enabled
					}
					break;
				case CPU::disabled_status:
					if (!currentThread->hasPendingShutdown()) {
						ThreadManager::addIdler(currentThread);
						ThreadManager::switchThreads(currentThread, nullptr);
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
	
};


#endif // CPU_ACTIVATION_HPP
