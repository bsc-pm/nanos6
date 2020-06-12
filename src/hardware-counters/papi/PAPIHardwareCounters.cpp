/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <iostream>

#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <papi.h>

#include "PAPIHardwareCounters.hpp"
#include "PAPIThreadHardwareCounters.hpp"
#include "PAPITaskHardwareCounters.hpp"
#include "PAPICPUHardwareCounters.hpp"


int PAPIHardwareCounters::_idMap[HWCounters::HWC_PAPI_NUM_EVENTS];
int PAPIHardwareCounters::_numEnabledCounters = 0;

static void readAndResetPAPICounters(
	PAPIThreadHardwareCounters *papiThreadCounters,
	long long *countersBuffer
) {
	int ret;
	int eventSet = papiThreadCounters->getEventSet();

	ret = PAPI_read(eventSet, countersBuffer);
	FatalErrorHandler::failIf(
		ret != PAPI_OK,
		ret, " when reading a PAPI event set",
		PAPI_strerror(ret)
	);
	ret = PAPI_reset(eventSet);
	FatalErrorHandler::failIf(
		ret != PAPI_OK,
		ret, " when resetting a PAPI event set",
		PAPI_strerror(ret)
	);
}

PAPIHardwareCounters::PAPIHardwareCounters(
	bool verbose,
	const std::string &,
	std::vector<HWCounters::counters_t> &enabledEvents
) {
	int ret;

	/* Initialize the library */
	ret = PAPI_library_init(PAPI_VER_CURRENT);
	FatalErrorHandler::failIf(
		ret != PAPI_VER_CURRENT,
		ret, " when initializing the PAPI library: ",
		PAPI_strerror(ret)
	);
	ret = PAPI_thread_init((unsigned long (*)(void)) (pthread_self));
	FatalErrorHandler::failIf(
		ret != PAPI_OK,
		ret, " when initializing the PAPI library for threads: ",
		PAPI_strerror(ret)
	);
	ret = PAPI_set_domain(PAPI_DOM_USER);
	FatalErrorHandler::failIf(
		ret != PAPI_OK,
		ret, " when setting the default PAPI domain to user only: ",
		PAPI_strerror(ret)
	);

	for (unsigned short i = 0; i < HWCounters::HWC_PAPI_NUM_EVENTS; ++i)
		_idMap[i] = DISABLED_PAPI_COUNTER;

	int cnt = 0;
	for (unsigned short i = 0; i < enabledEvents.size(); ++i) {
		int code;
		short id = enabledEvents[i];
		if (id >= HWCounters::HWC_PAPI_MIN_EVENT && id <= HWCounters::HWC_PAPI_MAX_EVENT) {
			ret = PAPI_event_name_to_code((char *) HWCounters::counterDescriptions[id], &code);
			FatalErrorHandler::failIf(
				ret != PAPI_OK,
				ret, std::string(" PAPI ") + HWCounters::counterDescriptions[id] + " event translation not found",
				PAPI_strerror(ret)
			);
			ret = PAPI_query_event(code);
			FatalErrorHandler::failIf(
				ret != PAPI_OK,
				ret, std::string("PAPI ") + HWCounters::counterDescriptions[id] + " not available on this machine",
				PAPI_strerror(ret)
			);
			_enabledPAPIEventCodes.push_back(code);
			_idMap[id - HWCounters::HWC_PAPI_MIN_EVENT] = cnt++;
		}
	}

	_numEnabledCounters = _enabledPAPIEventCodes.size();
	if (!_numEnabledCounters) {
		FatalErrorHandler::warnIf(true,
			"No PAPI events enabled, disabling hardware counters"
		);
		_enabled = false;
	} else {
		_enabled = true;
	}

	if (verbose)
		std::cout << "PAPI events enabled: " << _numEnabledCounters << std::endl;
}

void PAPIHardwareCounters::threadInitialized(ThreadHardwareCountersInterface *threadCounters)
{
	int ret;
	int eventSet = PAPI_NULL;
	PAPIThreadHardwareCounters *papiThreadCounters = (PAPIThreadHardwareCounters *) threadCounters;

	assert(papiThreadCounters != nullptr);

	if (!_enabled)
		return;

	/* Register the thread into PAPI */
	ret = PAPI_register_thread();
	FatalErrorHandler::failIf(
		ret != PAPI_OK,
		ret, " when registering a new thread into PAPI: ",
		PAPI_strerror(ret)
	);

	/* Create an EventSet */
	ret = PAPI_create_eventset(&eventSet);
	FatalErrorHandler::failIf(
		ret != PAPI_OK,
		ret, " when creating a PAPI event set: ",
		PAPI_strerror(ret)
	);

	/* Add Total Instructions Executed to our EventSet */
	ret = PAPI_add_events(eventSet,
			      _enabledPAPIEventCodes.data(),
			      _enabledPAPIEventCodes.size());
	FatalErrorHandler::failIf(
		ret != PAPI_OK,
		ret, " when initializing the PAPI event set of a new thread: ",
		PAPI_strerror(ret)
	);

	papiThreadCounters->setEventSet(eventSet);

	/* Start counting */
	ret = PAPI_start(eventSet);
	FatalErrorHandler::failIf(
		ret != PAPI_OK,
		ret, " when starting a PAPI event set",
		PAPI_strerror(ret)
	);
}

void PAPIHardwareCounters::threadShutdown(ThreadHardwareCountersInterface *)
{
	int ret;

	if (!_enabled)
		return;

	// TODO How can we stop the counters without passing a buffer?
	//ret = PAPI_stop(papiThreadCounters->getEventSet(), counter);
	//FatalErrorHandler::failIf(
	//	ret != PAPI_OK,
	//	ret, " when stopping a PAPI event set",
	//	PAPI_strerror(ret)
	//);

	ret = PAPI_unregister_thread();
	FatalErrorHandler::failIf(
		ret != PAPI_OK,
		ret, " when unregistering a PAPI thread",
		PAPI_strerror(ret)
	);
}

void PAPIHardwareCounters::taskReinitialized(TaskHardwareCountersInterface *taskCounters)
{
	PAPITaskHardwareCounters *papiTaskCounters = (PAPITaskHardwareCounters *) taskCounters;
	assert(papiTaskCounters != nullptr);

	papiTaskCounters->clear();
}

void PAPIHardwareCounters::updateTaskCounters(
	ThreadHardwareCountersInterface *threadCounters,
	TaskHardwareCountersInterface *taskCounters
) {
	long long *countersBuffer;
	PAPITaskHardwareCounters   *papiTaskCounters   = (PAPITaskHardwareCounters *) taskCounters;
	PAPIThreadHardwareCounters *papiThreadCounters = (PAPIThreadHardwareCounters *) threadCounters;

	assert(papiTaskCounters != nullptr);
	assert(papiThreadCounters != nullptr);

	if (!_enabled)
		return;

	countersBuffer = papiTaskCounters->getCountersBuffer();
	readAndResetPAPICounters(papiThreadCounters, countersBuffer);
}

void PAPIHardwareCounters::updateRuntimeCounters(
	CPUHardwareCountersInterface *cpuCounters,
	ThreadHardwareCountersInterface *threadCounters
) {
	long long *countersBuffer;
	PAPICPUHardwareCounters    *papiCPUCounters    = (PAPICPUHardwareCounters *) cpuCounters;
	PAPIThreadHardwareCounters *papiThreadCounters = (PAPIThreadHardwareCounters *) threadCounters;

	assert(papiThreadCounters != nullptr);
	assert(papiCPUCounters != nullptr);

	if (!_enabled)
		return;

	countersBuffer = papiCPUCounters->getCountersBuffer();
	readAndResetPAPICounters(papiThreadCounters, countersBuffer);
}
