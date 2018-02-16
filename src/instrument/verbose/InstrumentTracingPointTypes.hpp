/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_STATS_TRACING_POINT_TYPES_HPP
#define INSTRUMENT_STATS_TRACING_POINT_TYPES_HPP


#include <string>
#include <vector>


namespace Instrument {
	struct tracing_point_type_t {
		std::string _description;
		std::vector<std::string> _valueDescriptions;
	};
}


#endif // INSTRUMENT_STATS_TRACING_POINT_TYPES_HPP
