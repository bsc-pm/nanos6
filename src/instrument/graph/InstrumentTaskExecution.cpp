/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "ExecutionSteps.hpp"
#include "InstrumentGraph.hpp"
#include "InstrumentTaskExecution.hpp"

#include <InstrumentInstrumentationContext.hpp>

#include <mutex>


namespace Instrument {
	using namespace Graph;
	
	
	void startTask(__attribute__((unused)) task_id_t taskId, InstrumentationContext const &context)
	{
		std::lock_guard<SpinLock> guard(_graphLock);
		enter_task_step_t *enterTaskStep = new enter_task_step_t(context);
		_executionSequence.push_back(enterTaskStep);
	}
	
	void endTask(__attribute__((unused)) task_id_t taskId, InstrumentationContext const &context)
	{
		std::lock_guard<SpinLock> guard(_graphLock);
		exit_task_step_t *exitTaskStep = new exit_task_step_t(context);
		_executionSequence.push_back(exitTaskStep);
	}
	
}
