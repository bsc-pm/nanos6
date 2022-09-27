/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_DISTRIBUTED_MEMORY_HPP
#define INSTRUMENT_OVNI_DISTRIBUTED_MEMORY_HPP


#include <cassert>

#include "instrument/api/InstrumentDistributedMemory.hpp"
#include "OvniTrace.hpp"

namespace Instrument {
	inline int isDistributedInstrumentEnabled()
	{
		return 1;
	}

	inline void setupDistributedMemoryEnvironment(
		__attribute__((unused)) const nanos6_distributed_instrument_info_t *info
	) {
		assert(info != nullptr);
		Ovni::procSetRank(info->rank, info->num_ranks);
	}
}

#endif // INSTRUMENT_OVNI_DISTRIBUTED_MEMORY_HPP
