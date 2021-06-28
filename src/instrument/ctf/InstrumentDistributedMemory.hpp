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
	inline int isDistributedInstrumentEnabled()
	{
		return true;
	}

	inline void setupDistributedMemoryEnvironment(
		const nanos6_distributed_instrument_info_t * info
	) {
		assert(info != nullptr);

		// This offset corresponds to the one stored in the clock object of
		// the metadata file in the CTF trace. It is used to synchronize
		// multiple Nanos6 runtime clocks, so that events happening at the
		// same time appear with the same time value when processed by
		// babeltrace2. Further information can be found in:
		// https://babeltrace.org/docs/v2.0/libbabeltrace2/group__api-tir-cs.html
		const int64_t offset = info->clock_offset.offset_ns;

		CTFAPI::CTFTrace &trace = CTFAPI::CTFTrace::getInstance();
		trace.setDistributedMemory(offset, info->rank, info->num_ranks);

		if (trace.isDistributedMemoryEnabled() && info->rank == 0) {
			trace.makeFinalTraceDirectory();
		}
	}
}

#endif // INSTRUMENT_CTF_DISTRIBUTED_MEMORY_HPP
