/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_WORKERTHREAD_HPP
#define INSTRUMENT_CTF_WORKERTHREAD_HPP


#include "../api/InstrumentWorkerThread.hpp"

#include "../support/InstrumentThreadLocalDataSupport.hpp"

namespace Instrument {

	inline void WorkerGetsTask()
	{
		ThreadLocalData &tld = getThreadLocalData();
		if (tld.isBusyWaiting) {
			tld.isBusyWaiting = false;
			tp_worker_exit_busy_wait();
		}
	}

	inline void WorkerBusyWaits()
	{
		ThreadLocalData &tld = getThreadLocalData();
		if (!tld.isBusyWaiting) {
			tld.isBusyWaiting = true;
			tp_worker_enter_busy_wait();
		}
	}
}

#endif // INSTRUMENT_CTF_WORKERTHREAD_HPP

