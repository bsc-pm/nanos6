/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_WORKERTHREAD_HPP
#define INSTRUMENT_CTF_WORKERTHREAD_HPP

#include <cassert>

#include "CTFTracepoints.hpp"
#include "ctfapi/CTFAPI.hpp"
#include "instrument/api/InstrumentWorkerThread.hpp"
#include "instrument/ctf/InstrumentCPULocalData.hpp"
#include "instrument/support/InstrumentThreadLocalDataSupport.hpp"

namespace Instrument {

	inline void workerThreadSpins()
	{
		CPULocalData *cpuLocalData = getCTFCPULocalData();
		CTFAPI::CTFStream *userStream = cpuLocalData->userStream;
		CTFAPI::CTFKernelStream *kernelStream = cpuLocalData->kernelStream;
		assert(kernelStream != nullptr);
		assert(userStream != nullptr);

		CTFAPI::flushCurrentVirtualCPUBufferIfNeeded(userStream, userStream);

		if (kernelStream) {
			CTFAPI::updateKernelEvents(kernelStream, userStream);
			CTFAPI::flushCurrentVirtualCPUBufferIfNeeded(kernelStream, userStream);
		}
	}

	inline void workerThreadObtainedTask()
	{
		ThreadLocalData &tld = getThreadLocalData();
		if (tld.isBusyWaiting) {
			tld.isBusyWaiting = false;
			tp_worker_exit_busy_wait();
		}
	}

	inline void workerThreadBusyWaits()
	{
		ThreadLocalData &tld = getThreadLocalData();
		if (!tld.isBusyWaiting) {
			tld.isBusyWaiting = true;
			tp_worker_enter_busy_wait();
		}
	}
}

#endif // INSTRUMENT_CTF_WORKERTHREAD_HPP

