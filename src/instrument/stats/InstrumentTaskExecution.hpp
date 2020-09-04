/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_STATS_TASK_EXECUTION_HPP
#define INSTRUMENT_STATS_TASK_EXECUTION_HPP

#include "InstrumentStats.hpp"
#include "instrument/api/InstrumentTaskExecution.hpp"
#include "instrument/support/InstrumentThreadLocalDataSupport.hpp"


namespace Instrument {
	inline void startTask(task_id_t, InstrumentationContext const &)
	{
	}

	inline void endTask(task_id_t, InstrumentationContext const &)
	{
	}

	inline void destroyTask(task_id_t taskId, InstrumentationContext const &)
	{
		assert(taskId->_currentTimer != nullptr);

		taskId->_currentTimer->stop();
		taskId->_currentTimer = nullptr;

		ThreadLocalData &threadLocal = getThreadLocalData();
		Instrument::Stats::PhaseInfo &phaseInfo = threadLocal._threadInfo.getCurrentPhaseRef();
		Instrument::Stats::TaskInfo &taskInfo = phaseInfo._perTask[taskId->_type];
		taskInfo += taskId->_times;

		delete taskId;
	}

	inline void startTaskforCollaborator(task_id_t, task_id_t, bool, InstrumentationContext const &)
	{
	}

	inline void endTaskforCollaborator(task_id_t, task_id_t, bool, InstrumentationContext const &)
	{
	}
}


#endif // INSTRUMENT_STATS_TASK_EXECUTION_HPP
