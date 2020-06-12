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
		HWC_PQOS_MIN_EVENT = 100,                       // PQOS: Minimum event id
		HWC_PQOS_MON_EVENT_L3_OCCUP = 100,              // PQOS: LLC Usage
		HWC_PQOS_MON_EVENT_LMEM_BW = 101,               // PQOS: Local Memory Bandwidth
		HWC_PQOS_MON_EVENT_RMEM_BW = 102,               // PQOS: Remote Memory Bandwidth
		HWC_PQOS_PERF_EVENT_LLC_MISS = 103,             // PQOS: LLC Misses
		HWC_PQOS_PERF_EVENT_RETIRED_INSTRUCTIONS = 104, // PQOS: Retired Instructions
		HWC_PQOS_PERF_EVENT_UNHALTED_CYCLES = 105,      // PQOS: Unhalted cycles
		HWC_PQOS_MAX_EVENT = 105,                       // PQOS: Maximum event id
		HWC_PQOS_NUM_EVENTS = HWC_PQOS_MAX_EVENT - HWC_PQOS_MIN_EVENT + 1,
		//    PAPI EVENTS    //
		HWC_PAPI_MIN_EVENT = 200,                       // PAPI: Minimum event id
		HWC_PAPI_TOT_INS = 200,                         // PAPI: Instructions completed
		HWC_PAPI_TOT_CYC = 201,                         // PAPI: Total Cycles
		HWC_PAPI_L1_LDM  = 202,                         // PAPI: Level 1 load misses
		HWC_PAPI_L1_STM  = 203,                         // PAPI: Level 1 store misses
		HWC_PAPI_MAX_EVENT = 203,
		HWC_PAPI_NUM_EVENTS = HWC_PAPI_MAX_EVENT - HWC_PAPI_MIN_EVENT + 1,
		//    GENERAL    //
		HWC_TOTAL_NUM_EVENTS = HWC_PQOS_NUM_EVENTS + HWC_PAPI_NUM_EVENTS
	};

	static std::map<uint64_t, const char* const> counterDescriptions = {
		{HWC_PQOS_MON_EVENT_L3_OCCUP,              "PQOS_MON_EVENT_L3_OCCUP"},
		{HWC_PQOS_MON_EVENT_LMEM_BW,               "PQOS_MON_EVENT_LMEM_BW"},
		{HWC_PQOS_MON_EVENT_RMEM_BW,               "PQOS_MON_EVENT_RMEM_BW"},
		{HWC_PQOS_PERF_EVENT_LLC_MISS,             "PQOS_PERF_EVENT_LLC_MISS"},
		{HWC_PQOS_PERF_EVENT_RETIRED_INSTRUCTIONS, "PQOS_PERF_EVENT_RETIRED_INSTRUCTIONS"},
		{HWC_PQOS_PERF_EVENT_UNHALTED_CYCLES,      "PQOS_PERF_EVENT_UNHALTED_CYCLES"},
		{HWC_PAPI_TOT_INS,                         "PAPI_TOT_INS"},
		{HWC_PAPI_TOT_CYC,                         "PAPI_TOT_CYC"},
		{HWC_PAPI_L1_LDM,                          "PAPI_L1_LDM"},
		{HWC_PAPI_L1_STM,                          "PAPI_L1_STM"}
	};
}

#endif // SUPPORTED_HARDWARE_COUNTERS_HPP
