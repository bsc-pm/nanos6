/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef SCHEDULER_HPP
#define SCHEDULER_HPP

#include <vector>

#include <InstrumentTaskStatus.hpp>
#include "lowlevel/FatalErrorHandler.hpp"
#include "tasks/Task.hpp"

#include "LeafScheduler.hpp"
#include "NodeScheduler.hpp"

class Scheduler {
	static std::vector<LeafScheduler *> _CPUScheduler;
	static NodeScheduler *_topScheduler;
	
public:
	static void initialize();

	static void shutdown();
	
	static inline ComputePlace *addReadyTask(Task *task, ComputePlace *computePlace, SchedulerInterface::ReadyTaskHint hint = SchedulerInterface::NO_HINT)
	{
		assert(task != nullptr);
		FatalErrorHandler::failIf(task->isTaskloop(), "Task loop not supported yet"); // TODO
		
		Instrument::taskIsReady(task->getInstrumentationTaskId());
		if (computePlace != nullptr) {
			_CPUScheduler[((CPU *)computePlace)->_virtualCPUId]->addTask(task, hint);
		} else {
			_CPUScheduler[0]->addTask(task, hint); // TODO: make sure this CPU is enabled
		}
		
		return nullptr;
	}
	
	
	static Task *getReadyTask(ComputePlace *computePlace, __attribute__((unused)) Task *currentTask, bool doWait = false)
	{
		assert(computePlace != nullptr);
		
		return _CPUScheduler[((CPU *)computePlace)->_virtualCPUId]->getTask(doWait);
	}
	
	static inline ComputePlace *getIdleComputePlace(__attribute__((unused)) bool force=false)
	{
		return nullptr;
	}
	
	static void disableComputePlace(ComputePlace *computePlace)
	{
		assert(computePlace != nullptr);
		
		_CPUScheduler[((CPU *)computePlace)->_virtualCPUId]->disable();
	}
	
	static void enableComputePlace(ComputePlace *computePlace)
	{
		assert(computePlace != nullptr);
		
		_CPUScheduler[((CPU *)computePlace)->_virtualCPUId]->enable();
	}
};


#endif // SCHEDULER_HPP
