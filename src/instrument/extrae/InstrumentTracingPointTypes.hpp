/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_TRACING_POINT_TYPES_HPP
#define INSTRUMENT_EXTRAE_TRACING_POINT_TYPES_HPP

#include "extrae_types.h"


namespace Instrument {
	struct tracing_point_type_t {
		extrae_type_t _type;
		
		tracing_point_type_t()
			: _type(~ ((extrae_type_t) 0) )
		{
		}
		
		bool operator<(tracing_point_type_t const &other) const
		{
			return (_type < other._type);
		}
	};
}


#endif // INSTRUMENT_EXTRAE_TRACING_POINT_TYPES_HPP
