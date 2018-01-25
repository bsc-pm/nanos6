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
		if (_sampleTaskCount) {
			extrae_combined_events_t ce;
			
			ce.HardwareCounters = 0;
			ce.Callers = 0;
			ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
			ce.nEvents = 2;
			ce.nCommunications = 0;
			
			ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
			ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
			
			ce.Types[0] = _readyTasksEventType;
			ce.Values[0] = (extrae_value_t) _readyTasks;
			ce.Types[1] = _liveTasksEventType;
			ce.Values[1] = (extrae_value_t) _liveTasks;
			
			if (_traceAsThreads) {
				_extraeThreadCountLock.readLock();
			}
			
			Extrae_emit_CombinedEvents ( &ce );
			
			if (_traceAsThreads) {
				_extraeThreadCountLock.readUnlock();
			}
		}
	}
	
}


#endif // INSTRUMENT_EXTRAE_LEADER_THREAD_HPP
