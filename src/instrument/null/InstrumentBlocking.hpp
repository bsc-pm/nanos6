/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_BLOCKING_HPP
#define INSTRUMENT_NULL_BLOCKING_HPP


#include "../api/InstrumentBlocking.hpp"


namespace Instrument {
	inline void enterBlocking(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void exitBlocking(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void unblockTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
}


#endif // INSTRUMENT_NULL_BLOCKING_HPP
