/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <papi.h>
#include <pthread.h>

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


void PAPIHardwareCounters::testMaximumNumberOfEvents()
{
	if (!_enabled) {
		return;
	}

	if (_verbose) {
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

			std::cout << "  - Enabling " << codeName << ": ";
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
	_verbose = verbose;
	for (size_t i = 0; i < HWCounters::HWC_PAPI_NUM_EVENTS; ++i) {
		_idMap[i] = DISABLED_PAPI_COUNTER;
	}

	// Initialize the library
	int ret = PAPI_library_init(PAPI_VER_CURRENT);
	if (ret != PAPI_VER_CURRENT) {
		FatalErrorHandler::fail(ret, " when initializing the PAPI library - ", PAPI_strerror(ret));
	}

	// TODO: Enable multiplex if too many events enabled?
	/*
	ret = PAPI_multiplex_init();
	if (ret != PAPI_OK) {
		FatalErrorHandler::fail(ret, " when initializing PAPI library multiplex support - ", PAPI_strerror(ret));
	}
	*/

	// Initialize the PAPI library for threads, and the domain
	ret = PAPI_thread_init(pthread_self);
	if (ret != PAPI_OK) {
		FatalErrorHandler::fail(ret, " when initializing the PAPI library for threads - ", PAPI_strerror(ret));
	}
	ret = PAPI_set_domain(PAPI_DOM_USER);
	if (ret != PAPI_OK) {
		FatalErrorHandler::fail(ret, " when setting the default PAPI domain to user only - ", PAPI_strerror(ret));
	}

	// Now test the availability of all the requested events
	if (_verbose) {
		std::cout << "-------------------------------------------------------" << std::endl;
		std::cout << "- Testing the availability of the requested PAPI events" << std::endl;
	}

	size_t innerId = 0;
	auto it = enabledEvents.begin();
	while (it != enabledEvents.end()) {
		HWCounters::counters_t id = *it;
		if (id >= HWCounters::HWC_PAPI_MIN_EVENT && id <= HWCounters::HWC_PAPI_MAX_EVENT) {
			if (_verbose) {
				std::cout << "  - Checking " << HWCounters::counterDescriptions[id] << ": ";
			}

			int code;
			ret = PAPI_event_name_to_code((char *) HWCounters::counterDescriptions[id], &code);
			if (ret != PAPI_OK) {
				FatalErrorHandler::fail(ret,
					HWCounters::counterDescriptions[id],
					" event not known by this version of PAPI - ",
					PAPI_strerror(ret)
				);
			}
			ret = PAPI_query_event(code);
			if (_verbose) {
				if (ret != PAPI_OK) {
					std::cout << "FAIL" << std::endl;
				} else {
					std::cout << "OK" << std::endl;
				}
			}
			if (ret != PAPI_OK) {
				FatalErrorHandler::warn(ret, " ",
					HWCounters::counterDescriptions[id],
					" event unknown in this version of PAPI, skipping it - ",
					PAPI_strerror(ret)
				);

				// Erase the event from the vector of enabled events
				it = enabledEvents.erase(it);
			} else {
				_enabledPAPIEventCodes.push_back(code);
				_idMap[id - HWCounters::HWC_PAPI_MIN_EVENT] = innerId++;
				++it;
			}
		}
	}

	_numEnabledCounters = _enabledPAPIEventCodes.size();
	if (!_numEnabledCounters) {
		FatalErrorHandler::warn("No PAPI events enabled, disabling this backend");
		_enabled = false;
	} else {
		_enabled = true;
	}

	// Test incompatibilities between PAPI events
	testMaximumNumberOfEvents();
	if (_verbose) {
		std::cout << "- Finished testing the availability of all the requested PAPI events" << std::endl;
		std::cout << "- Number of PAPI events enabled: " << _numEnabledCounters << std::endl;
		std::cout << "-------------------------------------------------------" << std::endl;
	}
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
		assert(papiThreadCounters != nullptr);
		assert(papiTaskCounters != nullptr);

		int eventSet = papiThreadCounters->getEventSet();
		papiTaskCounters->readCounters(eventSet);
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
		assert(papiThreadCounters != nullptr);

		int eventSet = papiThreadCounters->getEventSet();
		papiCPUCounters->readCounters(eventSet);
	}
}
