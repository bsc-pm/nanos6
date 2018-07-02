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
	EnvironmentVariable<size_t> _pollingIterations;
	
	std::atomic<size_t> _queueThreshold;
	std::atomic<bool> _rebalance;
	
	polling_slot_t _pollingSlot;
	SchedulerQueueInterface *_queue;
	
	NodeScheduler *_parent;
	ComputePlace *_computePlace;
	
	std::atomic<bool> _idle;
	
	SpinLock _globalLock;
	
	inline void handleQueueOverflow()
	{
		size_t th = _queueThreshold / 2;
		
		if (th == 0) {
			th = 1;
		}
		
		std::vector<Task *> taskBatch = _queue->getTaskBatch(th);
		if (taskBatch.size() > 0) {
			// queue might have been emptied just a moment ago
			_parent->addTaskBatch(this, taskBatch);
		}
	}

public:
	LeafScheduler(ComputePlace *computePlace, NodeScheduler *parent) :
		_pollingIterations("NANOS6_SCHEDULER_POLLING_ITER", 100000),
		_queueThreshold(0),
		_rebalance(false),
		_parent(parent),
		_computePlace(computePlace),
		_idle(false)
	{
		_queue = SchedulerQueueInterface::initialize();
		_parent->setChild(this);
	}
	
	~LeafScheduler()
	{
		delete _queue;
	}

	inline void addTask(Task *task, SchedulerInterface::ReadyTaskHint hint)
	{
		if (hint == SchedulerInterface::MAIN_TASK_HINT) {
			// This is the main task. Run here
			_pollingSlot.setTask(task);
			assert(!_idle);
		} else if (hint == SchedulerInterface::UNBLOCKED_TASK_HINT) {
			bool success;
			bool idle;
			
			{
				// Try to put it in the polling slot
				std::lock_guard<SpinLock> guard(_globalLock);
				success = _pollingSlot.setTask(task);
				idle = _idle;
			}
			
			if (success) {
				if (idle) {
					ThreadManager::resumeIdle((CPU *)_computePlace);
				}
			} else {
				size_t elements = _queue->addTask(task, hint);
				
				if (elements > _queueThreshold) {
					handleQueueOverflow();
				}
			}
		} else {
			// For ready tasks, addTask is always called from a thread in the
			// same CPU. Therefore, there is no need to check polling slots,
			// or to wake up any CPUs.
			assert(!_idle);
			
			size_t elements = _queue->addTask(task, hint);
			
			if (elements > _queueThreshold) {
				handleQueueOverflow();
			}
		}
		
		// Queue is already balanced
		_rebalance = false;
	}

	inline void addTaskBatch(__attribute__((unused)) SchedulerInterface *who, std::vector<Task *> &taskBatch)
	{
		assert(taskBatch.size() > 0);
		assert(who == _parent);
		
		Task *task = taskBatch.back();
		
		bool idle;
		bool success;
		
		{
			std::lock_guard<SpinLock> guard(_globalLock);
			success = _pollingSlot.setTask(task);
			idle = _idle;
		}
		
		if (success) {
			taskBatch.pop_back();
			if (idle) {
				ThreadManager::resumeIdle((CPU *)_computePlace);
			}
		}
		
		_queue->addTaskBatch(taskBatch);
	}
	
	inline Task *getTask(bool doWait)
	{
		Task *task;
		
		_idle = false;
		
		task = _pollingSlot.getTask();
		if (task != nullptr) {
			_rebalance = false;
			return task;
		}
		
		task = _queue->getTask();
		if (task != nullptr) {
			if (_rebalance) {
				bool expected = true;
				if (_rebalance.compare_exchange_strong(expected, false)) {
					if (_queue->getSize() > (_queueThreshold * 1.5)) {
						handleQueueOverflow();
					}
				}
			}
			
			return task;
		}
		
		_rebalance = false;
		
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
			// Mark as idle. Here, and somewhere else?
			std::lock_guard<SpinLock> guard(_globalLock);
			task = _pollingSlot.getTask();
			if (task == nullptr) {
				_idle = true;
			}
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
		
		Task *pollingTask = _pollingSlot.getTask();
		if (pollingTask != nullptr) {
			// A task may be added before the scheduler has been marked as non-idle in the parent
			taskBatch.push_back(pollingTask);
		}
		
		if (taskBatch.size() > 0) {
			_parent->addTaskBatch(this, taskBatch);
		}
	}
	
	inline void enable()
	{
	}
	
	inline void updateQueueThreshold(size_t queueThreshold)
	{
		if (queueThreshold < _queueThreshold) {
			_rebalance = true;
		}
		
		_queueThreshold = queueThreshold;
	}
};

#endif // LEAF_SCHEDULER_HPP
