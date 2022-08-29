/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_SCHEDULER_HPP
#define INSTRUMENT_OVNI_SCHEDULER_HPP

#include "../api/InstrumentScheduler.hpp"
#include "InstrumentTaskId.hpp"
#include "OvniTrace.hpp"

namespace Instrument {

	inline void enterAddReadyTask()
	{
		Ovni::addReadyTaskEnter();
	}

	inline void exitAddReadyTask()
	{
		Ovni::addReadyTaskExit();
	}

	inline void enterGetReadyTask()
	{
	}

	inline void exitGetReadyTask()
	{
	}

	inline void enterSchedulerLock()
	{
	}

	inline void schedulerLockBecomesServer()
	{
		Ovni::schedServerEnter();
	}

	inline void exitSchedulerLockAsClient(
		__attribute__((unused)) task_id_t taskId
	) {
		Ovni::schedReceiveTask();
	}

	inline void exitSchedulerLockAsClient()
	{
	}

	inline void schedulerLockServesTask(
		__attribute__((unused)) task_id_t taskId
	) {
		Ovni::schedAssignTask();
	}

	inline void exitSchedulerLockAsServer()
	{
		Ovni::schedServerExit();
	}

	inline void exitSchedulerLockAsServer(
		__attribute__((unused)) task_id_t taskId
	)
	{
		Ovni::schedSelfAssignTask();
		Ovni::schedServerExit();
	}

	inline void enterProcessReadyTasks()
	{
		Ovni::processReadyEnter();
	}

	inline void exitProcessReadyTasks()
	{
		Ovni::processReadyExit();
	}
}

#endif // INSTRUMENT_OVNI_SCHEDULER_HPP
