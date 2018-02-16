/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_TRACING_POINTS_HPP
#define INSTRUMENT_NULL_TRACING_POINTS_HPP


#include "../api/InstrumentTracingPoints.hpp"


namespace Instrument {
	inline void createNumericTracingPointType(
		/* OUT */ __attribute__((unused)) tracing_point_type_t &tracingPointType,
		__attribute__((unused)) std::string const &name,
		__attribute__((unused)) std::string const &description
	) {
	}
	
	
	inline void createScopeTracingPointType(
		/* OUT */ __attribute__((unused)) tracing_point_type_t &tracingPointType,
		__attribute__((unused)) std::string const &name,
		__attribute__((unused)) std::string const &startDescription,
		__attribute__((unused)) std::string const &endDescription
	) {
	}
	
	
	inline void createEnumeratedTracingPointTypePair(
		/* OUT */ __attribute__((unused)) tracing_point_type_t &tracingPointType,
		__attribute__((unused)) std::string const &name,
		__attribute__((unused)) std::vector<std::string> const &valueDescriptions
	) {
	}
	
	
	template<typename... TS>
	inline void trace(
		__attribute__((unused)) InstrumentationContext const &context /* = ThreadInstrumentationContext::getCurrent() */,
		__attribute__((unused)) TS... tracePointInstances)
	{
	}
}


#endif // INSTRUMENT_NULL_TRACING_POINTS_HPP
