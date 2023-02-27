/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2023 Barcelona Supercomputing Center (BSC)
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

	inline void workerIdle(bool)
	{
	}

	inline void workerThreadBusyWaits()
	{
		ThreadLocalData &tld = getThreadLocalData();
		if (!tld.isBusyWaiting) {
			tld.isBusyWaiting = true;
			tp_worker_enter_busy_wait();
		}
	}

	inline void workerThreadBegin()
	{
	}

	inline void workerThreadEnd()
	{
	}

	inline void enterHandleTask()
	{
	}

	inline void exitHandleTask()
	{
	}

	inline void enterSwitchTo() {}
	inline void exitSwitchTo() {}
	inline void enterSuspend() {}
	inline void exitSuspend() {}
	inline void enterResume() {}
	inline void exitResume() {}
}

#endif // INSTRUMENT_CTF_WORKERTHREAD_HPP

