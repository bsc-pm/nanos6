/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <pthread.h>
#include <papi.h>

#include "PAPICPUHardwareCounters.hpp"
#include "PAPIHardwareCounters.hpp"
#include "PAPITaskHardwareCounters.hpp"
#include "PAPIThreadHardwareCounters.hpp"
#include "hardware-counters/CPUHardwareCountersInterface.hpp"
#include "hardware-counters/TaskHardwareCountersInterface.hpp"
#include "hardware-counters/ThreadHardwareCountersInterface.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


size_t PAPIHardwareCounters::_numEnabledCounters(0);
int PAPIHardwareCounters::_idMap[HWCounters::HWC_PAPI_NUM_EVENTS];

//! \brief Private function only accessible from this file
static void readAndResetPAPICounters(
	PAPIThreadHardwareCounters *papiThreadCounters,
	long long *countersBuffer
) {
	assert(_enabled);
	assert(countersBuffer != nullptr);
	assert(papiThreadCounters != nullptr);

	int eventSet = papiThreadCounters->getEventSet();
	int ret = PAPI_read(eventSet, countersBuffer);
	if (ret != PAPI_OK) {
		FatalErrorHandler::fail(ret, " when reading a PAPI event set - ", PAPI_strerror(ret));
	}

	// Reset (clean) counters
	ret = PAPI_reset(eventSet);
	if (ret != PAPI_OK) {
		FatalErrorHandler::fail(ret, " when resetting a PAPI event set - ", PAPI_strerror(ret));
	}
}

void PAPIHardwareCounters::testMaximumNumberOfEvents()
{
	if (!_enabled) {
		return;
	}

	if (_verbose) {
		std::cout << "----------------------------------------------------------" << std::endl;
		std::cout << "- Testing if all the requested PAPI events can co-exist..." << std::endl;
	}

	// Register the thread into PAPI and create an event set
	int ret = PAPI_register_thread();
	if (ret != PAPI_OK) {
		FatalErrorHandler::fail(ret, " when registering the main thread into PAPI - ", PAPI_strerror(ret));
	}
	int eventSet = PAPI_NULL;
	ret = PAPI_create_eventset(&eventSet);
	if (ret != PAPI_OK) {
		FatalErrorHandler::fail(ret, " when creating a PAPI event set for the main thread - ", PAPI_strerror(ret));
	}

	// After creating the event set and registering the main thread into PAPI
	// for the purpose of testing, test if all enabled events can co-exist
	for (size_t i = 0; i < _enabledPAPIEventCodes.size(); ++i) {
		int eventCode = _enabledPAPIEventCodes[i];
		if (_verbose) {
			char codeName[PAPI_MAX_STR_LEN];
			ret = PAPI_event_code_to_name(eventCode, codeName);
			if (ret != PAPI_OK) {
				FatalErrorHandler::fail(ret, " when converting from PAPI code to PAPI event name - ", PAPI_strerror(ret));
			}

			std::cout << " - Enabling " << codeName << ": ";
		}

		// Try to add the event to the set
		ret = PAPI_add_event(eventSet, eventCode);
		if (_verbose) {
			if (ret != PAPI_OK) {
				std::cout << "FAIL" << std::endl;
			} else {
				std::cout << "OK" << std::endl;
			}
		}

		// Regardless of the verbosity, if it failed, abort the execution
		if (ret != PAPI_OK) {
			FatalErrorHandler::fail("Cannot simultaneously enable all the requested PAPI events due to incompatibilities");
		}
	}

	if (_verbose) {
		std::cout << "- Finished testing all the requested PAPI events" << std::endl;
		std::cout << "--------------------------------------------------------" << std::endl;
	}

	// Remove all the events from the EventSet, destroy it, and unregister the thread
	ret = PAPI_cleanup_eventset(eventSet);
	if (ret != PAPI_OK) {
		FatalErrorHandler::fail(ret, " when clearing the main thread's PAPI eventSet - ", PAPI_strerror(ret));
	}
	ret = PAPI_destroy_eventset(&eventSet);
	if (ret != PAPI_OK) {
		FatalErrorHandler::fail(ret, " when destorying the main thread's PAPI eventSet - ", PAPI_strerror(ret));
	}
	ret = PAPI_unregister_thread();
	if (ret != PAPI_OK) {
		FatalErrorHandler::fail(ret, " when unregistering the main thread from the PAPI library - ", PAPI_strerror(ret));
	}
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
	if (_enabled) {
		// Register the thread into PAPI and create an EventSet for it
		int ret = PAPI_register_thread();
		if (ret != PAPI_OK) {
			FatalErrorHandler::fail(ret, " when registering a new thread into PAPI - ", PAPI_strerror(ret));
		}
		int eventSet = PAPI_NULL;
		ret = PAPI_create_eventset(&eventSet);
		if (ret != PAPI_OK) {
			FatalErrorHandler::fail(ret, " when creating a PAPI event set - ", PAPI_strerror(ret));
		}

		// TODO: Remove? Keep?
		/*
		// Multiplex the EventSet
		ret = PAPI_set_multiplex(eventSet);
		if (ret != PAPI_OK) {
			FatalErrorHandler::fail(ret, " when enabling PAPI multiplex for a new thread - ", PAPI_strerror(ret));
		}
		*/

		// Add all the enabled events to the EventSet
		ret = PAPI_add_events(eventSet,
			_enabledPAPIEventCodes.data(),
			_enabledPAPIEventCodes.size()
		);
		if (ret != PAPI_OK) {
			FatalErrorHandler::fail(ret, " when initializing the PAPI event set of a new thread - ", PAPI_strerror(ret));
		}

		// Set the EventSet to the thread and start counting
		PAPIThreadHardwareCounters *papiThreadCounters = (PAPIThreadHardwareCounters *) threadCounters;
		assert(papiThreadCounters != nullptr);

		papiThreadCounters->setEventSet(eventSet);
		ret = PAPI_start(eventSet);
		if (ret != PAPI_OK) {
			FatalErrorHandler::fail(ret, " when starting a PAPI event set - ", PAPI_strerror(ret));
		}
	}
}

void PAPIHardwareCounters::threadShutdown(ThreadHardwareCountersInterface *)
{
	if (_enabled) {
		// TODO: How can we stop the counters without passing a buffer?
		/*
		assert(papiThreadCounters != nullptr);

		int ret = PAPI_stop(papiThreadCounters->getEventSet(), counter);
		if (ret != PAPI_OK) {
			FatalErrorHandler::fail(ret, " when stopping a PAPI event set - ", PAPI_strerror(ret));
		}
		*/

		int ret = PAPI_unregister_thread();
		if (ret != PAPI_OK) {
			FatalErrorHandler::fail(ret, " when unregistering a PAPI thread - ", PAPI_strerror(ret));
		}
	}
}

void PAPIHardwareCounters::taskReinitialized(TaskHardwareCountersInterface *taskCounters)
{
	if (_enabled) {
		PAPITaskHardwareCounters *papiTaskCounters = (PAPITaskHardwareCounters *) taskCounters;
		assert(papiTaskCounters != nullptr);

		papiTaskCounters->clear();
	}
}

void PAPIHardwareCounters::updateTaskCounters(
	ThreadHardwareCountersInterface *threadCounters,
	TaskHardwareCountersInterface *taskCounters
) {
	if (_enabled) {
		PAPIThreadHardwareCounters *papiThreadCounters = (PAPIThreadHardwareCounters *) threadCounters;
		PAPITaskHardwareCounters *papiTaskCounters = (PAPITaskHardwareCounters *) taskCounters;
		assert(papiTaskCounters != nullptr);

		long long *countersBuffer = papiTaskCounters->getCountersBuffer();
		readAndResetPAPICounters(papiThreadCounters, countersBuffer);
	}
}

void PAPIHardwareCounters::updateRuntimeCounters(
	CPUHardwareCountersInterface *cpuCounters,
	ThreadHardwareCountersInterface *threadCounters
) {
	if (_enabled) {
		PAPICPUHardwareCounters *papiCPUCounters = (PAPICPUHardwareCounters *) cpuCounters;
		PAPIThreadHardwareCounters *papiThreadCounters = (PAPIThreadHardwareCounters *) threadCounters;
		assert(papiCPUCounters != nullptr);

		long long *countersBuffer = papiCPUCounters->getCountersBuffer();
		readAndResetPAPICounters(papiThreadCounters, countersBuffer);
	}
}
