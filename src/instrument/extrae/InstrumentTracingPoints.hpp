/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_TRACING_POINTS_HPP
#define INSTRUMENT_EXTRAE_TRACING_POINTS_HPP


#include "../api/InstrumentTracingPoints.hpp"

#include "InstrumentExtrae.hpp"

#include <extrae_user_events.h>


namespace Instrument {
	inline void createNumericTracingPointType(
		/* OUT */ tracing_point_type_t &tracingPointType,
		__attribute__((unused)) std::string const &name,
		std::string const &description
	) {
		tracingPointType._type = _nextTracingPointKey++;
		
		if (!_initialized) {
			_delayedNumericTracingPoints.emplace(tracingPointType, description);
			return;
		}
		
		ExtraeAPI::define_event_type((extrae_type_t) EventType::TRACING_POINT_BASE + tracingPointType._type, description.c_str(), 0, nullptr, nullptr);
	}
	
	
	inline void createScopeTracingPointType(
		/* OUT */ tracing_point_type_t &tracingPointType,
		std::string const &name,
		std::string const &startDescription,
		std::string const &endDescription
	) {
		tracingPointType._type = _nextTracingPointKey++;
		
		if (!_initialized) {
			_delayedScopeTracingPoints.emplace(tracingPointType,scope_tracing_point_info_t(name, startDescription, endDescription));
			return;
		}
		
		extrae_value_t values[2] = {0, 1};
		char const *valueDescriptions[2];
		
		valueDescriptions[0] = startDescription.c_str();
		valueDescriptions[1] = endDescription.c_str();
		ExtraeAPI::define_event_type((extrae_type_t) EventType::TRACING_POINT_BASE + tracingPointType._type, name.c_str(), 2, values, valueDescriptions);
	}
	
	
	inline void createEnumeratedTracingPointTypePair(
		/* OUT */ tracing_point_type_t &tracingPointType,
		std::string const &name,
		std::vector<std::string> const &valueDescriptions
	) {
		tracingPointType._type = _nextTracingPointKey++;
		
		if (!_initialized) {
			_delayedEnumeratedTracingPoints.emplace(tracingPointType, enumerated_tracing_point_info_t(name, valueDescriptions));
			return;
		}
		
		extrae_value_t values[valueDescriptions.size()];
		char const *extraeValueDescriptions[valueDescriptions.size()];
		
		for (size_t i = 0; i < valueDescriptions.size(); i++) {
			values[i] = i;
			extraeValueDescriptions[i] = valueDescriptions[i].c_str();
		}
		
		ExtraeAPI::define_event_type((extrae_type_t) EventType::TRACING_POINT_BASE + tracingPointType._type, name.c_str(), valueDescriptions.size(), values, extraeValueDescriptions);
	}
	
	
	namespace Extrae {
		inline void fillEventTypeAndValues(__attribute__((unused)) extrae_type_t *types, __attribute__((unused)) extrae_value_t *values)
		{
		}
		
		template<typename... TS>
		inline void fillEventTypeAndValues(extrae_type_t *types, extrae_value_t *values, tracing_point_instance_t const &instance, TS... tracePointInstances)
		{
			types[0] = instance._tracingPointType._type;
			types[0] += (extrae_type_t) EventType::TRACING_POINT_BASE;
			types++;
			values[0] = instance._value;
			values++;
			fillEventTypeAndValues(types, values, tracePointInstances...);
		}
	}
	
	
	template<typename... TS>
	inline void trace(__attribute__((unused)) InstrumentationContext const &context /* = ThreadInstrumentationContext::getCurrent() */, TS... tracePointInstances)
	{
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 0;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = sizeof...(tracePointInstances);
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		Extrae::fillEventTypeAndValues(ce.Types, ce.Values, tracePointInstances...);
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
}


#endif // INSTRUMENT_EXTRAE_TRACING_POINTS_HPP
