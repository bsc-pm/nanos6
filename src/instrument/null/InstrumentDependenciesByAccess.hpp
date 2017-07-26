/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_DEPENDENCIES_BY_ACCESS_HPP
#define INSTRUMENT_NULL_DEPENDENCIES_BY_ACCESS_HPP



#include "../api/InstrumentDependenciesByAccess.hpp"
#include "dependencies/DataAccessType.hpp"


namespace Instrument {
	inline void registerTaskAccess(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) DataAccessType accessType,
		__attribute__((unused)) bool weak,
		__attribute__((unused)) void *start,
		__attribute__((unused)) size_t length,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
}


#endif // INSTRUMENT_NULL_DEPENDENCIES_BY_ACCESS_HPP
