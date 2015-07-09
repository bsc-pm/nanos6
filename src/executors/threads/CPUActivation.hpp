#ifndef CPU_ACTIVATION_HPP
#define CPU_ACTIVATION_HPP


#include <cassert>

#include "CPU.hpp"
#include "ThreadManager.hpp"


using namespace threaded_executor_internals;


class CPUActivation {
public:
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
		
		CPU *cpu = currentThread->getHardwarePlace();
		assert(cpu != nullptr);
		
		bool successful = false;
		
		while (!successful) {
			CPU::activation_status_t currentStatus = cpu->_activationStatus;
			switch (currentStatus) {
				case CPU::starting_status:
					successful = cpu->_activationStatus.compare_exchange_strong(currentStatus, CPU::enabled_status);
					assert(successful); // There should be no other thread changing this
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
						ThreadManager::yieldIdler(currentThread);
						successful = false; // Loop again, since the CPU has been enabled or is being enabled
					}
					break;
				case CPU::disabled_status:
					break;
			}
		}
	}
	
};


#endif // CPU_ACTIVATION_HPP
