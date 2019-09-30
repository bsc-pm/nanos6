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
		// Mark the task as ready prior to entering the lock
		Instrument::taskIsReady(task->getInstrumentationTaskId());
		HardwareCounters::stopTaskMonitoring(task);
		Monitoring::taskChangedStatus(task, ready_status);
		
		_instance->addReadyTask(task, computePlace, hint);
	}
	
	static inline Task *getReadyTask(ComputePlace *computePlace, ComputePlace *deviceComputePlace = nullptr)
	{
		return _instance->getReadyTask(computePlace, deviceComputePlace);
	}
	
	//! \brief Check if the scheduler has available work for the current CPU
	//!
	//! \param[in] computePlace The host compute place
	//! \param[in] deviceComputePlace The target device compute place if it exists
	static inline bool hasAvailableWork(ComputePlace *computePlace, ComputePlace *deviceComputePlace = nullptr)
	{
		return _instance->hasAvailableWork(computePlace, deviceComputePlace);
	}
	
};

#endif // SCHEDULER_HPP
