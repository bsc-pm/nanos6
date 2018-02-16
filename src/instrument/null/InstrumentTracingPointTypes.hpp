/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_TRACING_POINT_TYPES_HPP
#define INSTRUMENT_NULL_TRACING_POINT_TYPES_HPP


namespace Instrument {
	struct tracing_point_type_t {
		tracing_point_type_t()
		{
		}
		
		template <typename T>
		tracing_point_type_t(__attribute__((unused)) T const &anything)
		{
		}
	};
}


#endif // INSTRUMENT_NULL_TRACING_POINT_TYPES_HPP
