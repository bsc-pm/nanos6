/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_DISTRIBUTED_MEMORY_HPP
#define INSTRUMENT_NULL_DISTRIBUTED_MEMORY_HPP


#include "instrument/api/InstrumentDistributedMemory.hpp"


namespace Instrument {
	inline int isDistributedInstrumentEnabled() {
		return 0;
	}

	inline void setupDistributedMemoryEnvironment(
		__attribute__((unused)) const nanos6_distributed_instrument_info_t *info
	) {
	}
}

#endif // INSTRUMENT_NULL_DISTRIBUTED_MEMORY_HPP
