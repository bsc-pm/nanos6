/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef SCHEDULER_HPP
#define SCHEDULER_HPP

#include "SchedulerInterface.hpp"

#include <HardwareCounters.hpp>
#include <InstrumentTaskStatus.hpp>
#include <Monitoring.hpp>


class Scheduler {
	static SchedulerInterface *_instance;

public:
	static void initialize();
	static void shutdown();

	static inline void addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint = NO_HINT)
	{
		assert(computePlace == nullptr || computePlace->getType() == nanos6_host_device);

		// Mark the task as ready prior to entering the lock
		Instrument::taskIsReady(task->getInstrumentationTaskId());
		HardwareCounters::stopTaskMonitoring(task);
		Monitoring::taskChangedStatus(task, ready_status);

		_instance->addReadyTask(task, computePlace, hint);
	}

	static inline Task *getReadyTask(ComputePlace *computePlace, ComputePlace *deviceComputePlace = nullptr)
	{
		assert(computePlace == nullptr || computePlace->getType() == nanos6_host_device);
		return _instance->getReadyTask(computePlace, deviceComputePlace);
	}

	//! \brief Check if the scheduler has available work for the current CPU
	//!
	//! \param[in] computePlace The host compute place
	//! \param[in] deviceComputePlace The target device compute place if it exists
	static inline bool hasAvailableWork(ComputePlace *computePlace, ComputePlace *deviceComputePlace = nullptr)
	{
		assert(computePlace->getType() == nanos6_host_device);
		return _instance->hasAvailableWork(computePlace, deviceComputePlace);
	}

	//! \brief Notify the scheduler that a CPU is about to be disabled
	//! in case any tasks must be unassigned
	//!
	//! \param[in] cpuId The id of the cpu that will be disabled
	//! \param[in] task A task assigned to the current thread or nullptr
	//!
	//! \return Whether work was reassigned upon disabling this CPU
	static inline bool disablingCPU(size_t cpuId, Task *task)
	{
		return _instance->disablingCPU(cpuId, task);
	}

};

#endif // SCHEDULER_HPP
