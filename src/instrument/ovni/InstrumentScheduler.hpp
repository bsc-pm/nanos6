/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_SCHEDULER_HPP
#define INSTRUMENT_CTF_SCHEDULER_HPP

#include "../api/InstrumentScheduler.hpp"
#include "InstrumentTaskId.hpp"
#include "OVNITrace.hpp"

namespace Instrument {

	//! \brief Enters the scheduler addReadyTask method
	inline void enterAddReadyTask()
	{
		OVNI::schedSubmitEnter();
	}

	//! \brief Exits the scheduler addReadyTask method
	inline void exitAddReadyTask()
	{
		OVNI::schedSubmitExit();
	}

	//! \brief Enters the scheduler addReadyTask method
	inline void enterGetReadyTask()
	{
		// while busy waiting, the worker is continuously requesting
		// tasks. This will quicly fill the ctf buffer, so we disable
		// both the enter and exit while busywaing.

		ThreadLocalData &tld = getThreadLocalData();
		if (tld.hungry)
			return;

		tld.hungry = true;
		OVNI::schedHungry();
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
		OVNI::schedServerEnter();
	}

	inline void exitSchedulerLockAsClient(
		__attribute__((unused)) task_id_t taskId
	) {
		OVNI::schedRecv();
		OVNI::schedFill();

		ThreadLocalData &tld = getThreadLocalData();
		tld.hungry = false;
	}

	inline void exitSchedulerLockAsClient()
	{
	}

	inline void schedulerLockServesTask(
		__attribute__((unused)) task_id_t taskId
	) {
		OVNI::schedSend();
	}

	inline void exitSchedulerLockAsServer()
	{
		OVNI::schedServerExit();
	}

	inline void exitSchedulerLockAsServer(
		__attribute__((unused)) task_id_t taskId
	)
	{
		OVNI::schedSelfAssign();
		OVNI::schedServerExit();
		OVNI::schedFill();

		ThreadLocalData &tld = getThreadLocalData();
		tld.hungry = false;
	}
}

#endif // INSTRUMENT_CTF_SCHEDULER_HPP
