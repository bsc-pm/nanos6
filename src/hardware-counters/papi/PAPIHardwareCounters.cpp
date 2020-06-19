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

void PAPIHardwareCounters::testMaximumNumberOfEvents()
{
	int ret;
	int eventSet = PAPI_NULL;

	if (!_enabled)
		return;

	if (_verbose)
		std::cout << "Trying to enable simultaneously all the requested PAPI events" << std::endl;

	/* Register the thread into PAPI */
	ret = PAPI_register_thread();
	FatalErrorHandler::failIf(
		ret != PAPI_OK,
		ret, " when registering the main thread into PAPI: ",
		PAPI_strerror(ret)
	);

	/* Create a test EventSet */
	ret = PAPI_create_eventset(&eventSet);
	FatalErrorHandler::failIf(
		ret != PAPI_OK,
		ret, " when creating a PAPI event set for the main thread",
		PAPI_strerror(ret)
	);

	for (auto it = _enabledPAPIEventCodes.begin(); it != _enabledPAPIEventCodes.end(); it++) {
		int code = *it;

		if (_verbose) {
			char codeName[PAPI_MAX_STR_LEN];
			ret = PAPI_event_code_to_name(code, codeName);
			FatalErrorHandler::failIf(
				ret != PAPI_OK,
				ret, " when translating from PAPI code to PAPI event name",
				PAPI_strerror(ret)
			);
			std::cout << " - Enabling " << codeName << ": ";
		}

		ret = PAPI_add_event(eventSet, code);
		if (ret != PAPI_OK) {

			if (_verbose)
				std::cout << "FAIL" << std::endl;

			FatalErrorHandler::fail(
				"It was not possible to enable simultaneously all of the requested PAPI events. ",
				"Each processor has a finite number of hardware counter registers, some of them are general purpose (can track any hardware counter) and other are fixed (can only track a specific hardware counter). ",
				"Also, some counters are not compatible with some other. Therefore, it's not only a matter of which is the maximum number of hardware counters supported on a processor, it depens on the combination of requested counters. ",
				"Please, try reducing the number of requested counters and/or try another combination (note that the supplied order of PAPI events is not relevant).",
				"The PAPI \"papi_event_chooser\" tool might be of help in determining compatible sets of hardware counters"
			);
			break;

		} else if (_verbose) {
			std::cout << "OK" << std::endl;
		}
	}

	/* Remove all events from the eventset */
	ret = PAPI_cleanup_eventset(eventSet);
	FatalErrorHandler::failIf(
		ret != PAPI_OK,
		ret, " when clearing the main thread PAPI eventSet: ",
		PAPI_strerror(ret)
	);

	/* destroy the test EventSet */
	ret = PAPI_destroy_eventset(&eventSet);
	FatalErrorHandler::failIf(
		ret != PAPI_OK,
		ret, " when destorying the main thread PAPI eventSet: ",
		PAPI_strerror(ret)
	);

	/* Unregister this thread */
	ret = PAPI_unregister_thread();
	FatalErrorHandler::failIf(
		ret != PAPI_OK,
		ret, " when unregistering the main thread from the PAPI library: ",
		PAPI_strerror(ret)
	);
}

PAPIHardwareCounters::PAPIHardwareCounters(
	bool verbose,
	const std::string &,
	std::vector<HWCounters::counters_t> &enabledEvents
) {
	int ret;

	_verbose = verbose;

	/* Initialize the library */
	ret = PAPI_library_init(PAPI_VER_CURRENT);
	FatalErrorHandler::failIf(
		ret != PAPI_VER_CURRENT,
		ret, " when initializing the PAPI library: ",
		PAPI_strerror(ret)
	);
	// TODO enable multiplex if too many events enabled?
	//ret = PAPI_multiplex_init();
	//FatalErrorHandler::failIf(
	//	ret != PAPI_OK,
	//	ret, " when initializing PAPI library multiplex support: ",
	//	PAPI_strerror(ret)
	//);
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

	if (_verbose)
		std::cout << "Testing this processors' availability of requested events" << std::endl;

	int cnt = 0;
	for (auto it = enabledEvents.begin(), end = enabledEvents.end(); it != end; ++it) {
		int code;
		short id = *it;
		if (id >= HWCounters::HWC_PAPI_MIN_EVENT && id <= HWCounters::HWC_PAPI_MAX_EVENT) {

			if (_verbose)
				std::cout << " - Checking " << HWCounters::counterDescriptions[id] << ":";

			ret = PAPI_event_name_to_code((char *) HWCounters::counterDescriptions[id], &code);
			FatalErrorHandler::failIf(
				ret != PAPI_OK,
				ret, std::string(" PAPI ") + HWCounters::counterDescriptions[id] + " event not known by this version of PAPI: ",
				PAPI_strerror(ret)
			);
			ret = PAPI_query_event(code);
			if (ret != PAPI_OK) {

				if (_verbose) {
					std::cout << "Fail" << std::endl;
				}

				FatalErrorHandler::warn(
					ret, std::string(" PAPI ") + HWCounters::counterDescriptions[id] + " not available on this machine, skipping it: ",
					PAPI_strerror(ret)
				);

				enabledEvents.erase(it);
				end = enabledEvents.end();

				continue;
			}

			_enabledPAPIEventCodes.push_back(code);
			_idMap[id - HWCounters::HWC_PAPI_MIN_EVENT] = cnt++;

			if (_verbose)
				std::cout << " OK" << std::endl;
		}
	}

	_numEnabledCounters = _enabledPAPIEventCodes.size();
	if (!_numEnabledCounters) {
		FatalErrorHandler::warn("No PAPI events enabled, disabling hardware counters");
		_enabled = false;
	} else {
		_enabled = true;
	}

	testMaximumNumberOfEvents();

	if (_verbose)
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

	///* Multiplex the EventSet */
	//ret = PAPI_set_multiplex(eventSet);
	//FatalErrorHandler::failIf(
	//	ret != PAPI_OK,
	//	ret, " when enabling PAPI multiplex for a new thread: ",
	//	PAPI_strerror(ret)
	//);
	//

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
