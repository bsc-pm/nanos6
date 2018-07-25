/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef LEAF_SCHEDULER_HPP
#define LEAF_SCHEDULER_HPP

#include <vector>

#include "executors/threads/CPUManager.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "../../SchedulerInterface.hpp"

#include "NodeScheduler.hpp"
#include "TreeSchedulerInterface.hpp"
#include "TreeSchedulerQueueInterface.hpp"

class LeafScheduler: public TreeSchedulerInterface {
private:
	EnvironmentVariable<size_t> _pollingIterations;
	
	std::atomic<size_t> _queueThreshold;
	std::atomic<bool> _rebalance;
	
	SchedulerInterface::polling_slot_t _pollingSlot;
	TreeSchedulerQueueInterface *_queue;
	
	NodeScheduler *_parent;
	ComputePlace *_computePlace;
	
	std::atomic<bool> _idle;
	std::atomic<bool> _running;
	
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
			_parent->addTaskBatch(this, taskBatch, true);
		}
	}
	
	inline void forceRebalance()
	{
		std::vector<Task *> taskBatch = _queue->getTaskBatch(1);
		if (taskBatch.size() > 0) {
			_parent->addTaskBatch(this, taskBatch, false);
		}
	}


public:
	LeafScheduler(ComputePlace *computePlace, NodeScheduler *parent) :
		_pollingIterations("NANOS6_SCHEDULER_POLLING_ITER", 100000),
		_queueThreshold(0),
		_rebalance(false),
		_parent(parent),
		_computePlace(computePlace),
		_idle(false),
		_running(false)
	{
		_queue = TreeSchedulerQueueInterface::initialize();
		_parent->setChild(this);
	}
	
	~LeafScheduler()
	{
		delete _queue;
	}

	inline void addTask(Task *task, bool hasComputePlace, SchedulerInterface::ReadyTaskHint hint)
	{
		if (hasComputePlace) {
			// For ready tasks, addTask is always called from a thread in the
			// same CPU. Therefore, there is no need to check polling slots,
			// or to wake up any CPUs.
			assert(!_idle);
			assert(_running);
			
			size_t elements = _queue->addTask(task, hint);
			
			if (elements > _queueThreshold) {
				handleQueueOverflow();
			}
		} else {
			bool success = false;
			
			
			if (!_running) {
				bool idle;
				{
					// Try to put it in the polling slot
					std::lock_guard<SpinLock> guard(_globalLock);
					success = _pollingSlot.setTask(task);
					idle = _idle;
				}
			
				if (success && idle) {
					ThreadManager::resumeIdle((CPU *)_computePlace);
				}
			}
			
			if (!success) {
				size_t elements = _queue->addTask(task, hint);
				
				if (elements > _queueThreshold) {
					handleQueueOverflow();
				}
			}
		}
		
		// Queue is already balanced
		_rebalance = false;
	}

	inline void addTaskBatch(__attribute__((unused)) TreeSchedulerInterface *who, std::vector<Task *> &taskBatch, __attribute__((unused)) bool handleThreshold)
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
		
		_running = false;
		
		if (_idle) {
			_idle = false;
			CPUManager::unidleCPU((CPU *)_computePlace);
		}
		
		task = _pollingSlot.getTask();
		if (task != nullptr) {
			_rebalance = false;
			_running = true;
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
			
			_running = true;
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
			
			if (_queueThreshold != 0) {
				// Something weird is going on. Force rebalancing the rest of the system
				_parent->getTask(this, true);
			}
			
			std::lock_guard<SpinLock> guard(_globalLock);
			task = _pollingSlot.getTask();
			if (task == nullptr) {
				_idle = true;
				CPUManager::cpuBecomesIdle((CPU *)_computePlace);
			}
		}
		
		if (task != nullptr) {
			_running = true;
		}
		
		return task;
	}
	
	inline void disable()
	{
		if (_idle) {
			_idle = false;
			_parent->unidleChild(this);
			CPUManager::unidleCPU((CPU *)_computePlace);
		}
		
		std::vector<Task *> taskBatch = _queue->getTaskBatch(-1);
		
		Task *pollingTask = _pollingSlot.getTask();
		if (pollingTask != nullptr) {
			// A task may be added before the scheduler has been marked as non-idle in the parent
			taskBatch.push_back(pollingTask);
		}
		
		if (taskBatch.size() > 0) {
			_parent->addTaskBatch(this, taskBatch, true);
		}
		
		_parent->disableChild(this);
	}
	
	inline void enable()
	{
		_parent->enableChild(this);
	}
	
	inline void updateQueueThreshold()
	{
		// We ask our parent for the current value, so we get the most updated value.
		// If it was passed by parameter, it would be necessary to hold a lock while
		// updating the whole tree, to avoid having two different thresholds in
		// different scheduler nodes.
		
		size_t queueThreshold = _parent->getQueueThreshold();
		
		if (queueThreshold == 0) {
			if (_rebalance && _running) {
				// We may be causing starvation to other nodes
				forceRebalance();
			} else if (queueThreshold < _queueThreshold) {
				_rebalance = true;
			}
		} else if (queueThreshold < _queueThreshold) {
			// Rebalance later. Don't block the CPU that triggered the update
			_rebalance = true;
		}
		
		_queueThreshold = queueThreshold;
	}
};

#endif // LEAF_SCHEDULER_HPP
