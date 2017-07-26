/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_STATS_TASK_EXECUTION_HPP
#define INSTRUMENT_STATS_TASK_EXECUTION_HPP


#include "../api/InstrumentTaskExecution.hpp"
#include "../support/InstrumentThreadLocalDataSupport.hpp"

#include "InstrumentStats.hpp"


namespace Instrument {
	inline void startTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) InstrumentationContext const &context)
	{
	}
	
	inline void returnToTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) InstrumentationContext const &context)
	{
	}
	
	inline void endTask(__attribute__((unused)) task_id_t taskId, __attribute__((unused)) InstrumentationContext const &context)
	{
	}
	
	inline void destroyTask(task_id_t taskId, __attribute__((unused)) InstrumentationContext const &context)
	{
		assert(taskId->_currentTimer != 0);
		taskId->_currentTimer->stop();
		taskId->_currentTimer = 0;
		
		ThreadLocalData &threadLocal = getThreadLocalData();
		Instrument::Stats::PhaseInfo &phaseInfo = threadLocal._threadInfo.getCurrentPhaseRef();
		Instrument::Stats::TaskInfo &taskInfo = phaseInfo._perTask[taskId->_type];
		taskInfo += taskId->_times;
		taskInfo += taskId->_hardwareCounters;
		
		delete taskId;
	}
}


#endif // INSTRUMENT_STATS_TASK_EXECUTION_HPP
