/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_SCHEDULER_HPP
#define INSTRUMENT_CTF_SCHEDULER_HPP

#include "../api/InstrumentScheduler.hpp"
#include "CTFTracepoints.hpp"
#include "InstrumentTaskId.hpp"

namespace Instrument {

	//! \brief Enters the scheduler addReadyTask method
	inline void enterAddReadyTask()
	{
		tp_scheduler_add_task_enter();
	}

	//! \brief Exits the scheduler addReadyTask method
	inline void exitAddReadyTask()
	{
		tp_scheduler_add_task_exit();
	}

	//! \brief Enters the scheduler addReadyTask method
	inline void enterGetReadyTask()
	{
		// while busy waiting, the worker is continuously requesting
		// tasks. This will quicly fill the ctf buffer, so we disable
		// both the enter and exit while busywaing.

		ThreadLocalData &tld = getThreadLocalData();
		if (tld.isBusyWaiting)
			return;

		tp_scheduler_get_task_enter();
	}

	//! \brief Exits the scheduler addReadyTask method
	inline void exitGetReadyTask()
	{
		// see enterGetReadyTask comments above

		ThreadLocalData &tld = getThreadLocalData();
		if (tld.isBusyWaiting)
			return;

		tp_scheduler_get_task_exit();
	}

	inline void enterSchedulerLock()
	{
		ThreadLocalData &tld = getThreadLocalData();
		tld.schedulerLockTimestamp = CTFAPI::getRelativeTimestamp();
	}

	inline void schedulerLockBecomesServer()
	{
		ThreadLocalData &tld = getThreadLocalData();
		tp_scheduler_lock_server(tld.schedulerLockTimestamp);
	}

	inline void exitSchedulerLockAsClient(
		task_id_t taskId
	) {
		ThreadLocalData &tld = getThreadLocalData();
		tp_scheduler_lock_client(tld.schedulerLockTimestamp, taskId._taskId);
	}

	inline void exitSchedulerLockAsClient()
	{
		ThreadLocalData &tld = getThreadLocalData();
		if (!tld.isBusyWaiting) {
			tp_scheduler_lock_client(tld.schedulerLockTimestamp, 0);
		}
	}

	inline void schedulerLockServesTask(
		task_id_t taskId
	) {
		tp_scheduler_lock_assign(taskId._taskId);
	}

	inline void exitSchedulerLockAsServer()
	{
		tp_scheduler_lock_server_exit();
	}

}

#endif // INSTRUMENT_CTF_SCHEDULER_HPP
