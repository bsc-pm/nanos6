/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_DISTRIBUTED_MEMORY_HPP
#define INSTRUMENT_DISTRIBUTED_MEMORY_HPP


#include <nanos6.h>

#include "instrument/api/InstrumentDistributedMemory.hpp"

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>


namespace Instrument {
	//! This function is called (indirectly) from user space to check
	//! wheather the instrumentation backend is concerned about distributed
	//! memory tracing
	inline int isDistributedInstrumentEnabled();

	//! This function is called (indirectly) from user application code to
	//! enable distributted memory support in Nanos6 (probably from the main
	//! task). Instrumentation backends are suposed to perform the required
	//! initialization for distributed memory at this point. Backends can
	//! assume that a distributed memory barrier is called just after the
	//! Nanos6 API call.
	//! \param[in] info Structure describing details of the current process in the team
	inline void setupDistributedMemoryEnvironment(
		const nanos6_distributed_instrument_info_t *info
	);
}

#endif // INSTRUMENT_DISTRIBUTED_MEMORY_HPP
