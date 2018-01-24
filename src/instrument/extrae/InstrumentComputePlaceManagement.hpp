/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_COMPUTE_PLACE_MANAGEMENT_HPP
#define INSTRUMENT_EXTRAE_COMPUTE_PLACE_MANAGEMENT_HPP


#include "InstrumentComputePlaceId.hpp"
#include "InstrumentExtrae.hpp"
#include "../api/InstrumentComputePlaceManagement.hpp"

#include <alloca.h>


// This is not defined in the extrae headers
extern "C" void Extrae_change_num_threads (unsigned n);


namespace Instrument {
	inline compute_place_id_t createdCPU(unsigned int virtualCPUId)
	{
		if (!_traceAsThreads) {
			Extrae_change_num_threads(extrae_nanos_get_num_cpus_and_external_threads());
		}
		
		return compute_place_id_t(virtualCPUId);
	}
	
	inline compute_place_id_t createdGPU()
	{
		return compute_place_id_t();
	}
	
	
	inline void suspendingComputePlace(__attribute__((unused)) compute_place_id_t const &computePlace)
	{
		if (!_traceAsThreads) {
			extrae_combined_events_t ce;
			
			ce.HardwareCounters = 0;
			ce.Callers = 0;
			ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
			ce.nEvents = 1;
			ce.nCommunications = 0;
			
			ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
			ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
			
			ce.Types[0] = _runtimeState;
			ce.Values[0] = (extrae_value_t) NANOS_NOT_RUNNING;
			
			Extrae_emit_CombinedEvents ( &ce );
		}
	}
	
	inline void resumedComputePlace(__attribute__((unused)) compute_place_id_t const &computePlace)
	{
		if (!_traceAsThreads) {
			extrae_combined_events_t ce;
			
			ce.HardwareCounters = 0;
			ce.Callers = 0;
			ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
			ce.nEvents = 1;
			ce.nCommunications = 0;
			
			ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
			ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
			
			ce.Types[0] = _runtimeState;
			ce.Values[0] = (extrae_value_t) NANOS_IDLE;
			
			Extrae_emit_CombinedEvents ( &ce );
		}
	}
	
	inline void shuttingDownComputePlace(__attribute__((unused)) compute_place_id_t const &computePlace)
	{
		if (!_traceAsThreads) {
			extrae_combined_events_t ce;
			
			ce.HardwareCounters = 0;
			ce.Callers = 0;
			ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
			ce.nEvents = 1;
			ce.nCommunications = 0;
			
			ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
			ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
			
			ce.Types[0] = _runtimeState;
			ce.Values[0] = (extrae_value_t) NANOS_SHUTDOWN;
			
			Extrae_emit_CombinedEvents ( &ce );
		}
	}
}


#endif // INSTRUMENT_EXTRAE_COMPUTE_PLACE_MANAGEMENT_HPP
