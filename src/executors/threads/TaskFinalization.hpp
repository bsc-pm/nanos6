/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_FINALIZATION_HPP
#define TASK_FINALIZATION_HPP

#include <InstrumentComputePlaceId.hpp>
#include <InstrumentTaskExecution.hpp>
#include <InstrumentThreadId.hpp>

#include "hardware/places/ComputePlace.hpp"
#include "WorkerThread.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/ompss/SpawnFunction.hpp"
#include "tasks/Task.hpp"



class TaskFinalization {
public:
	static void disposeOrUnblockTask(Task *task, ComputePlace *computePlace, bool fromBusyThread = false);
	
};


#endif // TASK_FINALIZATION_HPP
