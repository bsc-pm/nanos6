/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_STATS_TASK_WAIT_HPP
#define INSTRUMENT_STATS_TASK_WAIT_HPP


#include "../api/InstrumentTaskWait.hpp"

#include "tasks/Task.hpp"

#include "InstrumentTaskExecution.hpp"
#include "InstrumentTaskId.hpp"
#include "InstrumentStats.hpp"

#include <atomic>


namespace Instrument {
	inline void enterTaskWait(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) char const *invocationSource,
		__attribute__((unused)) task_id_t if0TaskId,
		__attribute__((unused)) InstrumentationContext const &context)
	{
	}
	
	inline void exitTaskWait(
		task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context)
	{
		// If a spawned function, count the taskwait as a frontier between phases
		if (!taskId->_hasParent) {
			Instrument::Stats::_phasesSpinLock.writeLock();
			
			assert(Instrument::Stats::_currentPhase == (Instrument::Stats::_phaseTimes.size() - 1));
			Instrument::Stats::_phaseTimes.back().stop();
			Instrument::Stats::_phaseTimes.emplace_back(true);
			
			Instrument::Stats::_currentPhase++;
			
			Instrument::Stats::_phasesSpinLock.writeUnlock();
		}
		
		Instrument::returnToTask(taskId, context);
	}
	
}


#endif // INSTRUMENT_STATS_TASK_WAIT_HPP
