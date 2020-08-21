/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_SCHEDULER_SUBSYTEM_ENTRY_POINTS_HPP
#define INSTRUMENT_CTF_SCHEDULER_SUBSYTEM_ENTRY_POINTS_HPP

#include "CTFTracepoints.hpp"
#include "instrument/api/InstrumentSchedulerSubsystemEntryPoints.hpp"

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

}

#endif // INSTRUMENT_CTF_SCHEDULER_SUBSYTEM_ENTRY_POINTS_HPP
