/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_TRACING_POINTS_HPP
#define INSTRUMENT_TRACING_POINTS_HPP


#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>

#include <string>
#include <vector>

#include <InstrumentTracingPointTypes.hpp>


namespace Instrument {
	struct tracing_point_instance_t {
		tracing_point_type_t const &_tracingPointType;
		long _value;
		
		tracing_point_instance_t(tracing_point_type_t const &tracingPointType, long value)
			:_tracingPointType(tracingPointType), _value(value)
		{
		}
	};
	
	// Example:
	// 	tracing_point_type_t itTP;
	// 	Instrument::createNumericTracingPointType(itTP, "Iteration", "Iteration");
	void createNumericTracingPointType(
		/* OUT */ tracing_point_type_t &tracingPointType,
		std::string const &name,
		std::string const &description
	);
	
	// Example:
	// 	tracing_point_type_t memTP;
	// 	Instrument::createScopeTracingPointType(memTP, "Allocating memory", "Enter", "Exit");
	void createScopeTracingPointType(
		/* OUT */ tracing_point_type_t &tracingPointType,
		std::string const &name,
		std::string const &startDescription,
		std::string const &endDescription
	);
	
	// Example:
	// 	tracing_point_type_t ioTP;
	// 	Instrument::createEnumeratedTracingPointTypePair(
	// 		ioTP, "I/O Operation",
	// 		{
	// 			"None",
	// 			"Reading",
	// 			"Writing"
	// 		}
	// 	);
	void createEnumeratedTracingPointTypePair(
		/* OUT */ tracing_point_type_t &tracingPointType,
		std::string const &name,
		std::vector<std::string> const &valueDescriptions
	);
	
	
	// Example:
	// 	Instrument::trace(Instrument::ThreadInstrumentationContext::getCurrent(),
	// 		Instrument::tracing_point_instance_t(itTP, i),
	// 		Instrument::tracing_point_instance_t(ioTP, 0)
	// 	);
	template<typename... TS>
	void trace(InstrumentationContext const &context /* = ThreadInstrumentationContext::getCurrent() */, TS... tracePointInstances);
	
}


#endif // INSTRUMENT_TRACING_POINTS_HPP
