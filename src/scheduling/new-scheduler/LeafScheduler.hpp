/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef LEAF_SCHEDULER_HPP
#define LEAF_SCHEDULER_HPP

#include <vector>

#include "executors/threads/ThreadManager.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "NodeScheduler.hpp"
#include "SchedulerInterface.hpp"
#include "SchedulerQueueInterface.hpp"

class LeafScheduler: public SchedulerInterface {
private:
	EnvironmentVariable<size_t> _minQueueThreshold;
	EnvironmentVariable<size_t> _maxQueueThreshold;
	EnvironmentVariable<size_t> _pollingIterations;
	
	polling_slot_t _pollingSlot;
	SchedulerQueueInterface *_queue;
	
	NodeScheduler *_parent;
	ComputePlace *_computePlace;
	
	std::atomic<bool> _idle;
	
	inline void handleQueueOverflow()
	{
		std::vector<Task *> taskBatch = _queue->getTaskBatch(_maxQueueThreshold - _minQueueThreshold);
		_parent->addTaskBatch(taskBatch);
	}

public:
	LeafScheduler(ComputePlace *computePlace, NodeScheduler *parent) :
		_minQueueThreshold("NANOS6_SCHEDULER_QUEUE_MIN_THRESHOLD", 10),
		_maxQueueThreshold("NANOS6_SCHEDULER_QUEUE_MAX_THRESHOLD", 20),
		_pollingIterations("NANOS6_SCHEDULER_POLLING_ITER", 100000),
		_parent(parent),
		_computePlace(computePlace),
		_idle(false)
	{
		assert(_maxQueueThreshold >= _minQueueThreshold);
		
		_queue = SchedulerQueueInterface::initialize();
		_parent->setChild(this);
	}

	inline void addTask(Task *task, SchedulerInterface::ReadyTaskHint hint)
	{
		// addTask is always called from a thread in the same CPU. Therefore,
		// there is no need to check polling slots, or to wake up any CPUs.
		
		size_t elements = _queue->addTask(task, hint);		
		
		if (elements > _maxQueueThreshold) {
			handleQueueOverflow();
		}
	}

	inline void addTaskBatch(std::vector<Task *> &taskBatch)
	{
		assert(taskBatch.size() > 0);
		
		_pollingSlot.setTask(taskBatch.back());
		taskBatch.pop_back();

		if (_idle) {
			ThreadManager::resumeIdle((CPU *)_computePlace);
		}
		
		_queue->addTaskBatch(taskBatch);
	}
	
	inline Task *getTask(bool doWait)
	{
		Task *task;
		
		if (_idle) {
			_idle = false;
		}
		
		task = _pollingSlot.getTask();
		if (task != nullptr) {
			return task;
		}
		
		task = _queue->getTask();
		if (task != nullptr) {
			return task;
		}
		
		_parent->getTask(this);
		
		if (doWait) {
			unsigned int iterations = 0;
			// TODO: exit before iterations are completed (in case CPU is disabled, or runtime shuts down)
			while (task == nullptr && iterations < _pollingIterations) {
				task = _pollingSlot.getTask();
				++iterations;
			}
		} else {
			task = _pollingSlot.getTask();
		}
		
		if (task == nullptr) {
			// Timedout
			// Mark as idle. Where?
			_idle = true;
		}
		
		return task;
	}
	
	inline void disable()
	{
		if (_idle) {
			_idle = false;
			_parent->unidleChild(this);
		}
		
		std::vector<Task *> taskBatch = _queue->getTaskBatch(-1);
		_parent->addTaskBatch(taskBatch);
	}
	
	inline void enable()
	{
	}
};

#endif // LEAF_SCHEDULER_HPP
