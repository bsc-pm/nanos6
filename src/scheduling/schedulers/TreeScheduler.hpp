/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef TREE_SCHEDULER_HPP
#define TREE_SCHEDULER_HPP

#include <vector>

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>
#include <InstrumentTaskStatus.hpp>
#include "lowlevel/FatalErrorHandler.hpp"
#include "tasks/Task.hpp"

#include "tree-scheduler/LeafScheduler.hpp"
#include "tree-scheduler/NodeScheduler.hpp"

class TreeScheduler: public SchedulerInterface {
	std::vector<LeafScheduler *> _CPUScheduler;
	NodeScheduler *_topScheduler;
	std::atomic<size_t> _schedRRCounter;
	
public:
	TreeScheduler();
	~TreeScheduler();

	ComputePlace *addReadyTask(Task *task, ComputePlace *computePlace, SchedulerInterface::ReadyTaskHint hint, bool doGetIdle);
	
	Task *getReadyTask(ComputePlace *computePlace, Task *currentTask, bool canMarkAsIdle = true);
	
	ComputePlace *getIdleComputePlace(bool force = false);
	
	void disableComputePlace(ComputePlace *computePlace);
	void enableComputePlace(ComputePlace *computePlace);
	
	std::string getName() const;
};


#endif // TREE_SCHEDULER_HPP
