/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_STATS_TASK_WAIT_HPP
#define INSTRUMENT_STATS_TASK_WAIT_HPP

#include <atomic>

#include "InstrumentStats.hpp"
#include "InstrumentTaskExecution.hpp"
#include "InstrumentTaskId.hpp"
#include "instrument/api/InstrumentTaskWait.hpp"
#include "tasks/Task.hpp"


namespace Instrument {
	inline void enterTaskWait(
		task_id_t,
		char const *,
		task_id_t,
		bool,
		InstrumentationContext const &
	) {
	}

	inline void exitTaskWait(
		task_id_t taskId,
		bool,
		InstrumentationContext const &)
	{
		// If a spawned function, count the taskwait as a frontier between phases
		if (!taskId->_hasParent) {
			Instrument::Stats::_phasesSpinLock.writeLock();

			assert(Instrument::Stats::_currentPhase == (int)(Instrument::Stats::_phaseTimes.size() - 1));

			Instrument::Stats::_phaseTimes.back().stop();
			Instrument::Stats::_phaseTimes.emplace_back(true);

			Instrument::Stats::_currentPhase++;

			Instrument::Stats::_phasesSpinLock.writeUnlock();
		}
	}

}


#endif // INSTRUMENT_STATS_TASK_WAIT_HPP
