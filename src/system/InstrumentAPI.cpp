/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include <nanos6/instrument.h>
#include <InstrumentDistributedMemory.hpp>


extern "C" int nanos6_is_distributed_instrument_enabled(void)
{
	return Instrument::isDistributedInstrumentEnabled();
}

extern "C" void nanos6_setup_distributed_instrument(
	const nanos6_distributed_instrument_info_t *info
) {
	assert(info != nullptr);
	Instrument::setupDistributedMemoryEnvironment(info);
}
