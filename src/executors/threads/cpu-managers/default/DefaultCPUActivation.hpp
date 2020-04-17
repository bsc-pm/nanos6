/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEFAULT_CPU_ACTIVATION_HPP
#define DEFAULT_CPU_ACTIVATION_HPP

#include <cassert>

#include "executors/threads/CPU.hpp"
#include "executors/threads/CPUManager.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "hardware-counters/HardwareCounters.hpp"
#include "monitoring/Monitoring.hpp"
#include "scheduling/Scheduler.hpp"

#include <InstrumentComputePlaceManagement.hpp>


class DefaultCPUActivation {
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
			case CPU::disabled_status:
			case CPU::disabling_status:
			case CPU::shutdown_status:
				return false;
			case CPU::lent_status:
			case CPU::lending_status:
			case CPU::acquired_status:
			case CPU::acquired_enabled_status:
			case CPU::returned_status:
			case CPU::shutting_down_status:
				assert(false);
				return false;
		}

		assert("Unhandled CPU activation status" == nullptr);
		return false;
	}

	//! \brief Enable a CPU
	//!
	//! \param[in] systemCPUId The id of the CPU to enable
	//! Whether the enabling worked
	//!
	//! \return Whether the enabling was performed
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
				case CPU::lent_status:
				case CPU::lending_status:
				case CPU::acquired_status:
				case CPU::acquired_enabled_status:
				case CPU::returned_status:
				case CPU::shutting_down_status:
					// The CPU should never be in here in this implementation
					assert(false);
					return false;
				case CPU::enabled_status:
				case CPU::enabling_status:
					// Already enabled or enabling, no change
					successful = true;
					break;
				case CPU::disabled_status:
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::enabling_status);
					if (successful) {
						// Wake up a thread to allow the state change to progress
						ThreadManager::resumeIdle(cpu);
					}
					break;
				case CPU::disabling_status:
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::enabled_status);
					break;
				case CPU::shutdown_status:
					// If the runtime is shutting down, no change and return
					return false;
			}
		}

		return true;
	}

	//! \brief Disable a CPU
	//!
	//! \param[in] systemCPUId The id of the CPU to disable
	//!
	//! \return Whether the disabling was performed
	static inline bool disable(size_t systemCPUId)
	{
		CPU *cpu = CPUManager::getCPU(systemCPUId);
		assert(cpu != nullptr);

		bool successful = false;
		while (!successful) {
			CPU::activation_status_t currentStatus = cpu->getActivationStatus();
			switch (currentStatus) {
				case CPU::uninitialized_status:
				case CPU::lent_status:
				case CPU::lending_status:
				case CPU::acquired_status:
				case CPU::acquired_enabled_status:
				case CPU::returned_status:
				case CPU::shutting_down_status:
					// The CPU should never be in here in this implementation
					assert(false);
					return false;
				case CPU::enabled_status:
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::disabling_status);
					break;
				case CPU::enabling_status:
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::disabled_status);
					break;
				case CPU::disabled_status:
				case CPU::disabling_status:
					successful = true;
					break;
				case CPU::shutdown_status:
					// If the runtime is shutting down, no change and return
					return false;
			}
		}

		return true;
	}

	//! \brief Check and handle CPU activation transitions
	//! NOTE: This code must be run regularly from within WorkerThreads
	//!
	//! \param[in,out] currentThread The current thread that is running on the
	//! cpu to check transitions for
	//!
	//! \return The current activation status of the CPU
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
				case CPU::lent_status:
				case CPU::lending_status:
				case CPU::acquired_status:
				case CPU::acquired_enabled_status:
				case CPU::returned_status:
				case CPU::shutting_down_status:
					// The CPU should never be in here in this implementation
					assert(false);
					return currentStatus;
				case CPU::enabled_status:
					successful = true;
					break;
				case CPU::enabling_status:
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::enabled_status);
					if (successful) {
						currentStatus = CPU::enabled_status;
						Instrument::resumedComputePlace(cpu->getInstrumentationId());
						Monitoring::cpuBecomesActive(cpu->getIndex());
					}
					break;
				case CPU::disabled_status:
					// The CPU is disabled, the thread should be idle.
					// It is not needed to call Instrument/Monitoring here,
					// since this is an extreme case that should barely
					// happen and the thread directly becomes idle again
					ThreadManager::addIdler(currentThread);
					currentThread->switchTo(nullptr);
					break;
				case CPU::disabling_status:
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::disabled_status);
					if (successful) {
						 // Loop again, since things may have changed
						successful = false;

						HardwareCounters::updateRuntimeCounters();
						Monitoring::cpuBecomesIdle(cpu->getIndex());
						Instrument::threadWillSuspend(currentThread->getInstrumentationId(), cpu->getInstrumentationId());
						Instrument::suspendingComputePlace(cpu->getInstrumentationId());

						// The CPU is disabling, the thread becomes idle
						ThreadManager::addIdler(currentThread);
						currentThread->switchTo(nullptr);
					}
					break;
				case CPU::shutdown_status:
					// No change and return immediately
					return currentStatus;
			}
		}

		// Return the current status, whether there was a change
		return currentStatus;
	}

	//! \brief Notify to a CPU that the runtime is shutting down
	static inline void shutdownCPU(CPU *cpu)
	{
		assert(cpu != nullptr);

		bool successful = false;
		while (!successful) {
			CPU::activation_status_t currentStatus = cpu->getActivationStatus();
			switch (currentStatus) {
				case CPU::uninitialized_status:
				case CPU::lent_status:
				case CPU::lending_status:
				case CPU::acquired_status:
				case CPU::acquired_enabled_status:
				case CPU::returned_status:
				case CPU::shutting_down_status:
					// The CPU should never be in here in this implementation
					assert(false);
					return;
				case CPU::enabled_status:
				case CPU::disabled_status:
				case CPU::disabling_status:
					// If the CPU is enabled, disabling or disabled, simply
					// change the status to shutdown
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::shutdown_status);
					break;
				case CPU::enabling_status:
					// If the CPU is being enabled, change to shutdown and
					// notify that it has resumed for shutdown
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::shutdown_status);
					if (successful) {
						Instrument::resumedComputePlace(cpu->getInstrumentationId());
						Monitoring::cpuBecomesActive(cpu->getIndex());
					}
					break;
				case CPU::shutdown_status:
					// If the runtime is shutting down, no change and return
					return;
			}
		}
	}

};


#endif // DEFAULT_CPU_ACTIVATION_HPP
