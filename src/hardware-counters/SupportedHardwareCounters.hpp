/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef SUPPORTED_HARDWARE_COUNTERS_HPP
#define SUPPORTED_HARDWARE_COUNTERS_HPP


namespace HWCounters {

	enum backends_t {
		PAPI_BACKEND = 0,
		PQOS_BACKEND,
		NUM_BACKENDS
	};

	enum counters_t {
		//    PQOS EVENTS    //
		PQOS_MIN_EVENT = 0,           // PQOS: Minimum event id
		PQOS_MON_EVENT_L3_OCCUP = 0,  // PQOS: LLC Usage
		PQOS_PERF_EVENT_IPC = 1,      // PQOS: IPC
		PQOS_MON_EVENT_LMEM_BW = 2,   // PQOS: Local Memory Bandwidth
		PQOS_MON_EVENT_RMEM_BW = 3,   // PQOS: Remote Memory Bandwidth
		PQOS_PERF_EVENT_LLC_MISS = 4, // PQOS: LLC Misses
		PQOS_MAX_EVENT = 4,           // PQOS: Maximum event id
		PQOS_NUM_EVENTS = 5,          // PQOS: Number of events
		//    PAPI EVENTS    //
		PAPI_MIN_EVENT = 5,           // PAPI: Minimum event id
		PAPI_PLACEHOLDER = 5,         // PAPI: TODO Remove when PAPI is integrated
		PAPI_MAX_EVENT = 5,           // PAPI: Maximum event id
		PAPI_NUM_EVENTS = 1,          // PAPI: Number of events
		//    GENERAL    //
		TOTAL_NUM_EVENTS = 6
	};

	char const * const counterDescriptions[TOTAL_NUM_EVENTS] = {
		//    PQOS EVENTS    //
		"PQOS_MON_EVENT_L3_OCCUP",
		"PQOS_PERF_EVENT_IPC",
		"PQOS_MON_EVENT_LMEM_BW",
		"PQOS_MON_EVENT_RMEM_BW",
		"PQOS_PERF_EVENT_LLC_MISS",
		//    PAPI EVENTS    //
		"PAPI_PLACEHOLDER"
	};

}

#endif // SUPPORTED_HARDWARE_COUNTERS_HPP
