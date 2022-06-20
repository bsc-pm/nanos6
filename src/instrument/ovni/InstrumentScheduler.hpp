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

	//! \brief Enters the scheduler addReadyTask method
	inline void enterAddReadyTask()
	{
		Ovni::schedSubmitEnter();
	}

	//! \brief Exits the scheduler addReadyTask method
	inline void exitAddReadyTask()
	{
		Ovni::schedSubmitExit();
	}

	//! \brief Enters the scheduler addReadyTask method
	inline void enterGetReadyTask()
	{
		// While busy waiting, the worker is continuously requesting
		// tasks. This will quickly fill the event buffer, so we disable
		// both the enter and exit while busywaing.

		ThreadLocalData &tld = getThreadLocalData();
		if (tld._hungry)
			return;

		tld._hungry = true;
		Ovni::schedHungry();
	}

	//! \brief Exits the scheduler addReadyTask method
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
		Ovni::schedFill();

		ThreadLocalData &tld = getThreadLocalData();
		tld._hungry = false;
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
		Ovni::schedFill();

		ThreadLocalData &tld = getThreadLocalData();
		tld._hungry = false;
	}
}

#endif // INSTRUMENT_OVNI_SCHEDULER_HPP
