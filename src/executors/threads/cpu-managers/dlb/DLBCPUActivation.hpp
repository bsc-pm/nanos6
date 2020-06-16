/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef DLB_CPU_ACTIVATION_HPP
#define DLB_CPU_ACTIVATION_HPP

#include <atomic>
#include <cassert>
#include <ctime>
#include <dlb.h>

#include "executors/threads/CPU.hpp"
#include "executors/threads/CPUManager.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware-counters/HardwareCounters.hpp"
#include "scheduling/Scheduler.hpp"

#include <InstrumentComputePlaceManagement.hpp>
#include <Monitoring.hpp>


class DLBCPUActivation {

private:

	//! Amount of nanoseconds to delay if a CPU wants to be enabled but is
	//! being used by another process
	static timespec _delayCPUEnabling;

	//! The current number of OWNED active CPUs
	static std::atomic<size_t> _numActiveOwnedCPUs;

private:

	//! \brief Lend a CPU to another runtime or process
	//!
	//! \param[in] systemCPUId The id of the CPU to be lent
	//!
	//! \return The DLB error code returned by the call
	static inline int dlbLendCPU(size_t systemCPUId)
	{
		int ret = DLB_LendCpu(systemCPUId);
		if (ret != DLB_SUCCESS) {
			FatalErrorHandler::fail("DLB Error ", ret, " when trying to lend a CPU(", systemCPUId, ")");
		}

		return ret;
	}

	//! \brief Reclaim a previously lent CPU
	//!
	//! \param[in] systemCPUId The id of the CPU to be reclaimed
	//! \param[in] ignoreError Whether to disregard an error in the operation
	//!
	//! \return The DLB error code returned by the call
	static inline int dlbReclaimCPU(size_t systemCPUId, bool ignoreError = false)
	{
		int ret = DLB_ReclaimCpu(systemCPUId);
		if (!ignoreError) {
			if (ret != DLB_NOUPDT && ret != DLB_NOTED && ret != DLB_SUCCESS) {
				FatalErrorHandler::fail("DLB Error ", ret, " when trying to reclaim a CPU(", systemCPUId, ")");
			}
		}

		return ret;
	}

	//! \brief Reclaim previously lent CPUs
	//!
	//! \param[in] numCPUs The number of CPUs to reclaim
	//!
	//! \return The DLB error code returned by the call
	static inline int dlbReclaimCPUs(size_t numCPUs)
	{
		int ret = DLB_ReclaimCpus(numCPUs);
		if (ret != DLB_NOUPDT && ret != DLB_NOTED && ret != DLB_SUCCESS) {
			FatalErrorHandler::fail("DLB Error ", ret, " when trying to reclaim ", numCPUs, " CPUs");
		}

		return ret;
	}

	//! \brief Reclaim previously lent CPUs
	//!
	//! \param[in] maskCPUs A mask signaling which CPUs to reclaim
	//!
	//! \return The DLB error code returned by the call
	static inline int dlbReclaimCPUs(const cpu_set_t &maskCPUs)
	{
		int ret = DLB_ReclaimCpuMask(&(maskCPUs));
		if (ret != DLB_NOUPDT && ret != DLB_NOTED && ret != DLB_SUCCESS) {
			FatalErrorHandler::fail("DLB Error ", ret, " when reclaiming CPUs through a mask");
		}

		return ret;
	}

	//! \brief Try to acquire a specific number of CPUs
	//!
	//! \param[in] numCPUs The number of CPUs to acquire
	//!
	//! \return The DLB error code returned by the call
	static inline int dlbAcquireCPUs(size_t numCPUs)
	{
		int ret = DLB_AcquireCpus(numCPUs);
		if (ret != DLB_NOUPDT && ret != DLB_NOTED && ret != DLB_SUCCESS) {
			FatalErrorHandler::fail("DLB Error ", ret, " when trying to acquire ", numCPUs, " CPUs");
		}

		return ret;
	}

	//! \brief Try to acquire a specific number of CPUs
	//!
	//! \param[in] maskCPUs A mask signaling which CPUs to acquire
	//!
	//! \return The DLB error code returned by the call
	static inline int dlbAcquireCPUs(const cpu_set_t &maskCPUs)
	{
		// We may get DLB_ERR_PERM if we're trying to acquire unregistered CPUs
		int ret = DLB_AcquireCpuMask(&maskCPUs);
		if (ret != DLB_NOUPDT  &&
			ret != DLB_NOTED   &&
			ret != DLB_SUCCESS &&
			ret != DLB_ERR_PERM
		) {
			FatalErrorHandler::fail("DLB Error ", ret, " when trying to acquire CPUs through a mask");
		}

		return ret;
	}

	//! \brief Try to return a CPU
	//!
	//! \param[in] systemCPUId The id of the CPU to be returned
	//!
	//! \return The DLB error code returned by the call
	static inline int dlbReturnCPU(size_t systemCPUId)
	{
		// We may get "DLB_ERR_PERM" only if the runtime is shutting down
		int ret = DLB_ReturnCpu(systemCPUId);
		if (ret != DLB_SUCCESS && ret != DLB_NOUPDT) {
			CPU *cpu = CPUManager::getCPU(systemCPUId);
			if ((ret != DLB_ERR_PERM) ||
				(ret == DLB_ERR_PERM && cpu->getActivationStatus() != CPU::shutdown_status)
			) {
				FatalErrorHandler::fail("DLB Error ", ret, " when trying to return a CPU(", systemCPUId, ")");
			}
		}

		return ret;
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
			case CPU::acquired_status:
			case CPU::acquired_enabled_status:
				return true;
			case CPU::uninitialized_status:
			case CPU::lent_status:
			case CPU::lending_status:
			case CPU::returned_status:
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
		FatalErrorHandler::warn(
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
		FatalErrorHandler::warn(
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

		size_t currentActiveCPUs = _numActiveOwnedCPUs.load();
		bool successful = false;

		// Always avoid lending the last owned CPU
		while (!successful && currentActiveCPUs > 1) {
			successful = _numActiveOwnedCPUs.compare_exchange_strong(currentActiveCPUs, currentActiveCPUs - 1);
			if (successful) {
				// NOTE: The CPU should only be lent when no work is available
				// Thus, we take for granted that the status of the CPU is
				// 'enabled_status' (unless it is shutting down)
				CPU::activation_status_t expectedStatus = CPU::enabled_status;
				bool successfulStatus = cpu->getActivationStatus().compare_exchange_strong(expectedStatus, CPU::lending_status);
				if (successfulStatus) {
					// NOTE: Lend the CPU and check if work was added while
					// lending it. This is to avoid a race condition between
					// adding tasks and idling CPUs. If a task is added after
					// we lend the CPU, try to acquire CPUs through DLB, and
					// the CPU we've just lent should be re-acquired
					dlbLendCPU(cpu->getSystemCPUId());

					HardwareCounters::cpuBecomesIdle();
					Monitoring::cpuBecomesIdle(cpu->getSystemCPUId());
					Instrument::suspendingComputePlace(cpu->getInstrumentationId());

					if (!Scheduler::isServingTasks()) {
						// No CPU is serving tasks in the scheduler, change to enabling,
						// call dlbReclaimCPU to let DLB know that we want to use this
						// CPU again and continue executing with the current thread.
						// If we are the first to reclaim the CPU, the reclaim will set
						// the CPU as ours within DLB and the thread will keep
						// executing until checkCPUStatusTransitions.
						// However, if another thread tries to acquire this CPU while
						// we're reclaiming, that acquire call will trigger an enable
						// callback, but as the status is lending, that callback will
						// loop until the new status is enabling and then do nothing,
						// thus it is safe to keep executing with the current thread
						expectedStatus = CPU::lending_status;
						successfulStatus = cpu->getActivationStatus().compare_exchange_strong(expectedStatus, CPU::enabling_status);
						assert(successfulStatus);

						dlbReclaimCPU(cpu->getSystemCPUId());
					} else {
						WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
						assert(currentThread != nullptr);

						// Change the status since the lending was successful
						expectedStatus = CPU::lending_status;
						successfulStatus = cpu->getActivationStatus().compare_exchange_strong(expectedStatus, CPU::lent_status);
						assert(successfulStatus);

						// This thread becomes idle
						ThreadManager::addIdler(currentThread);
						currentThread->switchTo(nullptr);
					}

					// NOTE: When a thread is resumed in the CPU we've just lent, the
					// CPU may not be available yet. We check this through DLB in
					// 'checkCPUStatusTransitions'
				} else {
					// The status change should always work, as noted previously
					assert(cpu->getActivationStatus() == CPU::shutdown_status);
					return;
				}
			}
		}
	}

	//! \brief Try to reclaim a CPU
	//!
	//! \param[in] systemCPUId The CPU's id
	static inline void reclaimCPU(size_t systemCPUId)
	{
		CPU *cpu = CPUManager::getCPU(systemCPUId);
		assert(cpu != nullptr);
		assert(cpu->isOwned());

		bool successful = false;
		while (!successful) {
			CPU::activation_status_t currentStatus = cpu->getActivationStatus();
			switch (currentStatus) {
				case CPU::lent_status:
					// Try to reclaim a CPU disregarding an error. We disregard any error
					// since the CPU may be reclaimed by the time we issue the request
					dlbReclaimCPU(systemCPUId, true);
					return;
				case CPU::disabled_status:
				case CPU::disabling_status:
				case CPU::uninitialized_status:
				case CPU::acquired_status:
				case CPU::acquired_enabled_status:
				case CPU::returned_status:
					assert(false);
					return;
				case CPU::lending_status:
					// In this case we iterate until the lending is complete
					break;
				case CPU::enabled_status:
				case CPU::enabling_status:
				case CPU::shutdown_status:
				case CPU::shutting_down_status:
					// The CPU should already be ours, abort the reclaim
					return;
			}
		}
	}

	//! \brief Try to acquire a specific number of external CPUs
	//!
	//! \param[in] numCPUs The number of CPUs to acquire
	static inline void acquireCPUs(size_t numCPUs)
	{
		assert(numCPUs > 0);

		dlbAcquireCPUs(numCPUs);
	}

	//! \brief Try to acquire a specific number of external CPUs
	//!
	//! \param[in] maskCPUS The mask of CPUs to acquire
	static inline void acquireCPUs(const cpu_set_t &maskCPUs)
	{
		assert(CPU_COUNT(&maskCPUs) > 0);

		dlbAcquireCPUs(maskCPUs);
	}

	//! \brief Return an external CPU
	//!
	//! \param[in,out] cpu The CPU to return
	static inline void returnCPU(CPU *cpu)
	{
		assert(cpu != nullptr);
		assert(!cpu->isOwned());

		HardwareCounters::cpuBecomesIdle();
		Monitoring::cpuBecomesIdle(cpu->getSystemCPUId());
		Instrument::suspendingComputePlace(cpu->getInstrumentationId());

		// Change the status to returned and return it
		CPU::activation_status_t expectedStatus = CPU::acquired_enabled_status;
		bool successful = cpu->getActivationStatus().compare_exchange_strong(expectedStatus, CPU::returned_status);
		if (successful) {
			// Since we do not want to keep the CPU, we use lend instead of return
			__attribute__((unused)) int ret = dlbLendCPU(cpu->getSystemCPUId());
			assert(ret == DLB_SUCCESS);

			WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
			assert(currentThread != nullptr);

			ThreadManager::addIdler(currentThread);
			currentThread->switchTo(nullptr);
		} else {
			assert(cpu->getActivationStatus() == CPU::shutdown_status);
		}
	}

	//! \brief DLB-specific callback called when a CPU is returned to us
	//!
	//! \param[in] systemCPUId The id of the CPU that can be used once again
	//! \param[in] args (Unused) DLB-compliant callback arguments
	static inline void dlbEnableCallback(int systemCPUId, void *)
	{
		CPU *cpu = CPUManager::getCPU(systemCPUId);
		assert(cpu != nullptr);

		bool successful = false;
		while (!successful) {
			CPU::activation_status_t currentStatus = cpu->getActivationStatus();
			switch (currentStatus) {
				case CPU::disabled_status:
				case CPU::disabling_status:
					// The CPU should never be in here in this implementation
					assert(false);
					return;
				case CPU::enabled_status:
				case CPU::enabling_status:
				case CPU::acquired_status:
				case CPU::acquired_enabled_status:
				case CPU::shutdown_status:
					// If we are not expecting a callback, ignore it
					successful = true;
					break;
				case CPU::lending_status:
					// In this case we iterate until the lending is complete
					break;
				case CPU::uninitialized_status:
					assert(!cpu->isOwned());

					// If the status is uninit, this CPU must be external
					// We have acquired, for the first time, a CPU we don't own
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::acquired_status);
					if (successful) {
						// Initialize the acquired CPU
						cpu->initialize();

						// Resume a thread so it sees the status change
						ThreadManager::resumeIdle(cpu, true);
					}
					break;
				case CPU::lent_status:
					assert(cpu->isOwned());

					// If we are expecting a callback, try to reenable the CPU
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::enabling_status);
					if (successful) {
						// Resume a thread so it sees the status changes
						ThreadManager::resumeIdle(cpu);
					}
					break;
				case CPU::returned_status:
					assert(!cpu->isOwned());

					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::acquired_status);
					if (successful) {
						// Resume a thread so it sees the status change
						ThreadManager::resumeIdle(cpu);
					}
					break;
				case CPU::shutting_down_status:
					assert(cpu->isOwned());

					// If the CPU was lent and the runtime is shutting down
					// reenable the CPU without changing the status
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::shutdown_status);
					if (successful) {
						Instrument::resumedComputePlace(cpu->getInstrumentationId());
						Monitoring::cpuBecomesActive(cpu->getSystemCPUId());
						ThreadManager::resumeIdle(cpu, true, true);
					}
					break;
			}
		}
	}

	//! \brief DLB-specific callback called when a CPU is asked to be disabled
	//!
	//! \param[in] systemCPUId The id of the CPU to disable
	//! \param[in] args (Unused) DLB-compliant callback arguments
	static inline void dlbDisableCallback(int systemCPUId, void *)
	{
		// Only unowned CPUs should execute the following callback
		CPU *cpu = CPUManager::getCPU(systemCPUId);
		assert(cpu != nullptr);
		assert(!cpu->isOwned());

		bool successful = false;
		while (!successful) {
			CPU::activation_status_t currentStatus = cpu->getActivationStatus();
			switch (currentStatus) {
				// The CPU should never be in here in this implementation
				case CPU::uninitialized_status:
				case CPU::disabled_status:
				case CPU::disabling_status:
				// The CPU is unowned, cannot be in here
				case CPU::enabled_status:
				case CPU::enabling_status:
				case CPU::lent_status:
				case CPU::lending_status:
				// The CPU should be acquired, cannot be in here
				case CPU::returned_status:
				case CPU::shutting_down_status:
					assert(false);
					return;
				case CPU::acquired_status:
				case CPU::acquired_enabled_status:
					// If the callback is called at this point, it means the
					// CPU is checking if it should be returned, thus we can
					// assure no task is being executed and we can return it
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::returned_status);
					if (successful) {
						WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
						assert(currentThread != nullptr);
						assert(currentThread->getComputePlace() == cpu);
						assert(currentThread->getTask() == nullptr);

						// The thread becomes idle
						ThreadManager::addIdler(currentThread);
						currentThread->switchTo(nullptr);
					}
					break;
				case CPU::shutdown_status:
					// If the callback is called at this point, the runtime is
					// shutting down. Instead of returning the CPU, simply use
					// it to shutdown faster and DLB_Finalize will take care of
					// returning the CPU afterwards
					successful = true;
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
				case CPU::acquired_enabled_status:
				case CPU::shutdown_status:
					// If the CPU is enabled or shutdown, do nothing
					successful = true;
					break;
				case CPU::lent_status:
				case CPU::returned_status:
				case CPU::shutting_down_status:
					// If a thread is woken up in this CPU but it is not
					// supposed to be running, readd the thread as idle
					ThreadManager::addIdler(currentThread);
					currentThread->switchTo(nullptr);
					break;
				case CPU::enabling_status:
					assert(cpu->isOwned());

					// If the CPU is owned and enabled, DLB may have woken us
					// even though another process may be using our CPU still
					// Before completing the enable, check if this is the case
					if (DLB_CheckCpuAvailability(cpu->getSystemCPUId()) != DLB_SUCCESS) {
						// The CPU is not ready yet, sleep for a bit
						nanosleep(&_delayCPUEnabling, nullptr);
					} else {
						successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::enabled_status);
						if (successful) {
							// This owned CPU becomes active now
							++_numActiveOwnedCPUs;

							currentStatus = CPU::enabled_status;
							Instrument::resumedComputePlace(cpu->getInstrumentationId());
							Monitoring::cpuBecomesActive(cpu->getSystemCPUId());
						}
					}
					break;
				case CPU::acquired_status:
					successful = cpu->getActivationStatus().compare_exchange_strong(currentStatus, CPU::acquired_enabled_status);
					if (successful) {
						currentStatus = CPU::acquired_enabled_status;
						Instrument::resumedComputePlace(cpu->getInstrumentationId());
						Monitoring::cpuBecomesActive(cpu->getSystemCPUId());
					}
					break;
			}
		}

		// Return the current status, whether there was a change
		return currentStatus;
	}

	//! \brief Check whether the CPU has to be returned
	//! NOTE: This function must be run after executing a task and
	//! before checking the status transitions of the CPU
	//!
	//! \param[in] thread The WorkerThread is currently checking
	//! whether has to return the CPU it is running on
	static inline void checkIfMustReturnCPU(WorkerThread *thread)
	{
		CPU *cpu = thread->getComputePlace();
		assert(cpu != nullptr);

		// If the CPU is not owned check if it must be returned
		if (!cpu->isOwned() && cpu->getActivationStatus() != CPU::shutdown_status) {
			// At the start of the checkIfMustReturnCPU call, the CPU might
			// already be returned, so we switch here to idle and switch
			// back quickly to active if the CPU was not returned
			HardwareCounters::cpuBecomesIdle();
			Monitoring::cpuBecomesIdle(cpu->getSystemCPUId());
			Instrument::suspendingComputePlace(cpu->getInstrumentationId());

			int ret = dlbReturnCPU(cpu->getSystemCPUId());
			if (ret != DLB_SUCCESS) {
				// The CPU wasn't returned, so resume instrumentation. In case
				// it was returned, the appropriate thread will resume it
				Instrument::resumedComputePlace(cpu->getInstrumentationId());
				Monitoring::cpuBecomesActive(cpu->getSystemCPUId());
			}
		}
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
				// The CPU should never be disabled in this implementation
				case CPU::disabled_status:
				case CPU::disabling_status:
				// The CPU should not already be shutting down
				case CPU::shutdown_status:
				case CPU::shutting_down_status:
					assert(false);
					return;
				case CPU::uninitialized_status:
					successful = true;
					break;
				case CPU::enabled_status:
				case CPU::enabling_status:
				case CPU::acquired_status:
				case CPU::acquired_enabled_status:
				case CPU::returned_status:
					// If the CPU is enabled, enabling or returned, changing to
					// shutdown is enough
					// If the CPU is acquired, switch to shutdown. Instead of
					// returning it, we use it a short amount of time to speed
					// up the shutdown process and DLB_Finalize will take care
					// of returning it
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
