/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_DATA_ACCESS_TYPES_HPP
#define INSTRUMENT_DATA_ACCESS_TYPES_HPP


#include <dependencies/DataAccessType.hpp>


namespace Instrument {
	enum access_object_type_t {
		regular_access_type,
		entry_fragment_type,
		taskwait_type,
		top_level_sink_type
	};
	
	
}


#endif // INSTRUMENT_DATA_ACCESS_TYPES_HPP
