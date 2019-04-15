/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef SUPPORTED_HARDWARE_COUNTERS_HPP
#define SUPPORTED_HARDWARE_COUNTERS_HPP

namespace HWCounters {
	enum counters_t {
		llc_usage = 0,
		ipc,
		local_mem_bandwidth,
		remote_mem_bandwidth,
		llc_miss_rate,
		// Add more counters in here if needed
		num_counters,
		invalid_counter = -1
	};
	
	char const * const counterDescriptions[num_counters] = {
		"Last Level Cache Usage (KB)",
		"Instructions Per Cycle",
		"Local Socket Memory Bandwidth (KB)",
		"Remote Sockets Memory Bandwidth (KB)",
		"Last Level Cache Miss Rate"
	};
	
}

#endif // SUPPORTED_HARDWARE_COUNTERS_HPP
