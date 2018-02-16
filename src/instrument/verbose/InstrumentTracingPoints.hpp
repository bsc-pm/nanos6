/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_STATS_TRACING_POINTS_HPP
#define INSTRUMENT_STATS_TRACING_POINTS_HPP


#include "InstrumentTracingPointTypes.hpp"
#include "../api/InstrumentTracingPoints.hpp"

#include "InstrumentVerbose.hpp"


namespace Instrument {
	inline void createNumericTracingPointType(
		/* OUT */ tracing_point_type_t &tracingPointType,
		__attribute__((unused)) std::string const &name,
		std::string const &description
	) {
		tracingPointType._description = description;
	}
	
	
	inline void createScopeTracingPointType(
		/* OUT */ tracing_point_type_t &tracingPointType,
		__attribute__((unused)) std::string const &name,
		__attribute__((unused)) std::string const &startDescription,
		__attribute__((unused)) std::string const &endDescription
	) {
		tracingPointType._description = name;
		tracingPointType._valueDescriptions.push_back(startDescription);
		tracingPointType._valueDescriptions.push_back(endDescription);
	}
	
	
	inline void createEnumeratedTracingPointTypePair(
		/* OUT */ tracing_point_type_t &tracingPointType,
		__attribute__((unused)) std::string const &name,
		__attribute__((unused)) std::vector<std::string> const &valueDescriptions
	) {
		tracingPointType._description = name;
		tracingPointType._valueDescriptions = valueDescriptions;
	}
	
	
	namespace Verbose {
		template<typename... TS>
		inline void fillTracingPointInstances(LogEntry *logEntry, tracing_point_instance_t const &instance, TS... tracePointInstances)
		{
			if (instance._tracingPointType._valueDescriptions.empty()) {
				logEntry->_contents << " " << instance._tracingPointType._description << ": " << instance._value;
			} else {
				logEntry->_contents << " " << instance._tracingPointType._description << ": " << instance._tracingPointType._valueDescriptions[instance._value];
			}
			fillTracingPointInstances(logEntry, tracePointInstances...);
		}
		
		
		inline void fillTracingPointInstances(__attribute__((unused)) LogEntry *logEntry)
		{
		}
	}
	
	
	template<typename... TS>
	inline void trace(
		__attribute__((unused)) InstrumentationContext const &context /* = ThreadInstrumentationContext::getCurrent() */,
		__attribute__((unused)) TS... tracePointInstances)
	{
		Instrument::Verbose::LogEntry *logEntry = Instrument::Verbose::getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		
		if (context._taskId != task_id_t()) {
			logEntry->_contents << " Task:" << context._taskId;
		}
		
		Instrument::Verbose::fillTracingPointInstances(logEntry, tracePointInstances...);
		
		addLogEntry(logEntry);
	}
}


#endif // INSTRUMENT_STATS_TRACING_POINTS_HPP
