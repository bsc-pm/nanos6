/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_LEADER_THREAD_HPP
#define INSTRUMENT_EXTRAE_LEADER_THREAD_HPP


#include "../api/InstrumentLeaderThread.hpp"
#include "InstrumentExtrae.hpp"


namespace Instrument {
	inline void leaderThreadSpin()
	{
		// Non-precise task count (sampled)
		if (_detailLevel < 1) {
			extrae_combined_events_t ce;
			
			ce.HardwareCounters = 0;
			ce.Callers = 0;
			ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
			ce.nEvents = 2;
			ce.nCommunications = 0;
			
			ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
			ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
			
			ce.Types[0] = (extrae_type_t) EventType::READY_TASKS;
			ce.Values[0] = (extrae_value_t) _readyTasks;
			ce.Types[1] = (extrae_type_t) EventType::LIVE_TASKS;
			ce.Values[1] = (extrae_value_t) _liveTasks;
			
			
			// These counters are not so reliable, so try to skip underflows
			if (((signed long long) ce.Values[0]) < 0) {
				ce.Values[0] = 0;
			}
			if (((signed long long) ce.Values[1]) < 0) {
				ce.Values[1] = 0;
			}
			
			if (_traceAsThreads) {
				_extraeThreadCountLock.readLock();
			}
			
			ExtraeAPI::emit_CombinedEvents ( &ce );
			
			if (_traceAsThreads) {
				_extraeThreadCountLock.readUnlock();
			}
		}
	}
	
}


#endif // INSTRUMENT_EXTRAE_LEADER_THREAD_HPP
