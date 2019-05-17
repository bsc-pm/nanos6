/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "ExecutionSteps.hpp"
#include "InstrumentGraph.hpp"
#include "InstrumentTaskWait.hpp"

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentTaskExecution.hpp>

#include <cassert>


namespace Instrument {
	using namespace Graph;
	
	
	void enterTaskWait(task_id_t taskId, char const *invocationSource, task_id_t if0TaskId, InstrumentationContext const &context)
	{
		std::lock_guard<SpinLock> guard(_graphLock);
		task_info_t &taskInfo = _taskToInfoMap[taskId];
		
		taskwait_id_t taskwaitId = _nextTaskwaitId++;
		taskwait_t *taskwait = new taskwait_t(taskwaitId, invocationSource, if0TaskId);
		taskwait->_task = taskId;
		taskwait->_taskPhaseIndex = taskInfo._phaseList.size();
		_taskwaitToInfoMap[taskwaitId] = taskwait;
		
		// Save the taskwait identifier in the current phase
		if (!taskInfo._phaseList.empty()) {
			phase_t *currentPhase = taskInfo._phaseList.back();
			currentPhase->_nextTaskwaitId = taskwaitId;
		}
		
		enter_taskwait_step_t *enterTaskwaitStep = new enter_taskwait_step_t(context, taskwaitId);
		taskInfo._phaseList.push_back(taskwait);
		_executionSequence.push_back(enterTaskwaitStep);
	}
	
	
	void exitTaskWait(task_id_t taskId, InstrumentationContext const &context)
	{
		std::lock_guard<SpinLock> guard(_graphLock);
		task_info_t &taskInfo = _taskToInfoMap[taskId];
		
		assert(!taskInfo._phaseList.empty());
		phase_t *taskwaitPhase = taskInfo._phaseList.back();
		taskwait_t *taskwait = dynamic_cast<taskwait_t *> (taskwaitPhase);
		assert(taskwait != nullptr);
		taskwait_id_t taskwaitId = taskwait->_taskwaitId;
		
		exit_taskwait_step_t *exitTaskwaitStep = new exit_taskwait_step_t(context, taskwaitId);
		_executionSequence.push_back(exitTaskwaitStep);
		
		// Instead of calling to Instrument::returnToTask we later on reuse the exitTaskwaitStep to also reactivate the task
	}
	
}
