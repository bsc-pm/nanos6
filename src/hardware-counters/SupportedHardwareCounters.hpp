/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef SUPPORTED_HARDWARE_COUNTERS_HPP
#define SUPPORTED_HARDWARE_COUNTERS_HPP

#include <map>


namespace HWCounters {

	enum backends_t {
		PAPI_BACKEND = 0,
		PQOS_BACKEND,
		RAPL_BACKEND,
		NUM_BACKENDS
	};

	// NOTE: To add new events, do as follows:
	// - If the event added is from an existing backend (e.g. PQOS):
	// -- 1) Add the new event before the current maximum event (PQOS_MAX_EVENT)
	// -- 2) The identifier of the new event should be the previous maximum event + 1
	// -- 3) Update PQOS_MAX_EVENT
	// -- WARNING: Be aware not to create collisions between backend identifiers!
	//
	// - If the event added is from a new backend:
	// -- 1) Add the new backend to the previous enum (backends_t)
	// -- 2) Following the observed pattern, add "MIN", "MAX", and "NUM" variables for
	//       the new backend, as well as any needed event identifier
	// -- 3) The identifiers should start with the previous minimum event backend + 100
	//       (i.e., NEWBACKEND_MIN_EVENT = PAPI_MIN_EVENT (200) + 100
	// -- 4) Add the MAX variable to the total number of events
	//       (i.e., TOTAL_NUM_EVENTS = ... + NEWBACKEND_NUM_EVENTS
	//
	// In all cases: Add a description of the event below (counterDescriptions)
	enum counters_t {
		//    PQOS EVENTS    //
		PQOS_MIN_EVENT = 100,                       // PQOS: Minimum event id
		PQOS_MON_EVENT_L3_OCCUP = 100,              // PQOS: LLC Usage
		PQOS_MON_EVENT_LMEM_BW = 101,               // PQOS: Local Memory Bandwidth
		PQOS_MON_EVENT_RMEM_BW = 102,               // PQOS: Remote Memory Bandwidth
		PQOS_PERF_EVENT_LLC_MISS = 103,             // PQOS: LLC Misses
		PQOS_PERF_EVENT_RETIRED_INSTRUCTIONS = 104, // PQOS: Retired Instructions
		PQOS_PERF_EVENT_UNHALTED_CYCLES = 105,      // PQOS: Unhalted cycles
		PQOS_MAX_EVENT = 105,                       // PQOS: Maximum event id
		PQOS_NUM_EVENTS = PQOS_MAX_EVENT - PQOS_MIN_EVENT + 1,
		//    PAPI EVENTS    //
		PAPI_MIN_EVENT = 200,                       // PAPI: Minimum event id
		PAPI_PLACEHOLDER = 200,                     // PAPI: TODO Remove when PAPI is integrated
		PAPI_MAX_EVENT = 200,                       // PAPI: Maximum event id
		PAPI_NUM_EVENTS = PAPI_MAX_EVENT - PAPI_MIN_EVENT + 1,
		//    GENERAL    //
		TOTAL_NUM_EVENTS = PQOS_NUM_EVENTS + PAPI_NUM_EVENTS
	};

	static std::map<uint64_t, const char* const> counterDescriptions = {
		{PQOS_MON_EVENT_L3_OCCUP,              "PQOS_MON_EVENT_L3_OCCUP"},
		{PQOS_MON_EVENT_LMEM_BW,               "PQOS_MON_EVENT_LMEM_BW"},
		{PQOS_MON_EVENT_RMEM_BW,               "PQOS_MON_EVENT_RMEM_BW"},
		{PQOS_PERF_EVENT_LLC_MISS,             "PQOS_PERF_EVENT_LLC_MISS"},
		{PQOS_PERF_EVENT_RETIRED_INSTRUCTIONS, "PQOS_PERF_EVENT_RETIRED_INSTRUCTIONS"},
		{PQOS_PERF_EVENT_UNHALTED_CYCLES,      "PQOS_PERF_EVENT_UNHALTED_CYCLES"},
		{PAPI_PLACEHOLDER,                     "PAPI_PLACEHOLDER"}
	};

}

#endif // SUPPORTED_HARDWARE_COUNTERS_HPP
