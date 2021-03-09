/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_DISTRIBUTED_MEMORY_HPP
#define INSTRUMENT_CTF_DISTRIBUTED_MEMORY_HPP


#include <cassert>

#include "instrument/api/InstrumentDistributedMemory.hpp"
#include "instrument/ctf/ctfapi/CTFTrace.hpp"


namespace Instrument {
	inline int isDistributedInstrumentEnabled() {
		CTFAPI::CTFTrace &trace = CTFAPI::CTFTrace::getInstance();
		return trace.getNumberOfRanks() != 0;
	}

	inline void setupDistributedMemoryEnvironment(
		const nanos6_distributed_instrument_info_t * info
	) {
		assert(info != nullptr);
		CTFAPI::CTFTrace &trace = CTFAPI::CTFTrace::getInstance();
		trace.setDistributedMemory(info->clock_offset.mean_sec,
					   info->rank, info->num_ranks);
		if (info->rank == 0) {
			trace.makeFinalTraceDirectory();
		}
	}
}

#endif // INSTRUMENT_CTF_DISTRIBUTED_MEMORY_HPP
