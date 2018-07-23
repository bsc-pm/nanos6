/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef NODE_SCHEDULER_HPP
#define NODE_SCHEDULER_HPP

#include <vector>

#include "executors/threads/ThreadManager.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "TreeSchedulerInterface.hpp"
#include "TreeSchedulerQueueInterface.hpp"

class NodeScheduler: public TreeSchedulerInterface {
private:
	EnvironmentVariable<size_t> _pollingIterations;
	
	std::atomic<size_t> _queueThreshold;
	std::atomic<bool> _rebalance;
	
	std::deque<TreeSchedulerInterface *> _idleChildren;
	TreeSchedulerQueueInterface *_queue;
	
	NodeScheduler *_parent;
	std::vector<TreeSchedulerInterface *> _children;
	std::atomic<size_t> _enabledChildren;
	
	SpinLock _globalLock;
	SpinLock _thresholdLock;
	
	inline void handleQueueOverflow(bool handleThreshold = true)
	{
		if (_parent != nullptr) {
			size_t elements = _queueThreshold / 2;
			
			if (elements == 0) {
				elements = 1;
			}
			
			std::vector<Task *> taskBatch = _queue->getTaskBatch(elements);
			if (taskBatch.size() > 0) {
				// queue might have been emptied just a moment ago
				_parent->addTaskBatch(this, taskBatch, handleThreshold);
			}
		} else {
			if (handleThreshold) {
				// Increase threshold and propagate downwards
				std::lock_guard<SpinLock> guard(_thresholdLock);
				size_t th = _queueThreshold * 2;
				
				if (th == 0) {
					th = 2;
				}
				
				updateQueueThreshold(th);
			}
		}
	}

public:
	NodeScheduler(NodeScheduler *parent = nullptr) :
		_pollingIterations("NANOS6_SCHEDULER_POLLING_ITER", 100000),
		_queueThreshold(0),
		_rebalance(false),
		_parent(parent),
		_enabledChildren(0)
	{
		_queue = TreeSchedulerQueueInterface::initialize();
		if (_parent != nullptr) {
			_parent->setChild(this);
		}
	}
	
	~NodeScheduler()
	{
		delete _queue;
		for (TreeSchedulerInterface *sched : _children) {
			delete sched;
		}
	}

	inline void addTaskBatch(TreeSchedulerInterface *who, std::vector<Task *> &taskBatch, bool handleThreshold)
	{
		TreeSchedulerInterface *idleChild = who;
		bool overflow = false;
		
		assert(who != nullptr);

		{
			std::lock_guard<SpinLock> guard(_globalLock);
			
			// If the caller is an idle child and is adding tasks, it is not idle anymore
			while (_idleChildren.size() > 0 && idleChild == who) {
				idleChild = _idleChildren.front();
				_idleChildren.pop_front();
			}
			
			if (idleChild == who) {
				size_t elements = _queue->addTaskBatch(taskBatch);
				if (elements > _queueThreshold) {
					overflow = true;
				}
			}
		}
		
		// Outside lock, call other nodes
		if (idleChild != who) {
			idleChild->addTaskBatch(this, taskBatch, handleThreshold);
		} else if (overflow) {
			handleQueueOverflow(handleThreshold);
		}
		
		// Queue is already balanced
		_rebalance = false;
	}
	
	inline void getTask(TreeSchedulerInterface *child, bool force = false)
	{
		size_t th = _queueThreshold;
		size_t elements = th / 2;
		
		if (elements == 0) {
			elements = 1;
		}
		
		std::vector<Task *> taskBatch;
	
		{
			std::lock_guard<SpinLock> guard(_globalLock);
			taskBatch = _queue->getTaskBatch(elements);
			
			if (taskBatch.size() == 0) {
				_idleChildren.push_back(child);
				_rebalance = false;
			}
		}
		
		// Outside lock, call other nodes
		if (taskBatch.size() > 0) {
			child->addTaskBatch(this, taskBatch, true);
		} else {
			if (_parent != nullptr) {
				_parent->getTask(this, force);
			}
		}
		
		if (_parent == nullptr) {
			if (force) {
				// Rare case, move threshold to 0. Hope this kicks a rebalance
				std::lock_guard<SpinLock> guard(_thresholdLock);
				updateQueueThreshold(0);
			} else {
				// Reduce threshold and propagate
				std::lock_guard<SpinLock> guard(_thresholdLock);
				updateQueueThreshold(_queueThreshold / 2);
			}
		} else if (_rebalance) {
			bool expected = true;
			if (_rebalance.compare_exchange_strong(expected, false)) {
				if (_queue->getSize() > (_queueThreshold * 1.5)) {
					handleQueueOverflow();
				}
			}
		}
	}
	
	inline void setChild(TreeSchedulerInterface *child)
	{
		++_enabledChildren;
		_children.push_back(child);
	}
	
	inline void unidleChild(TreeSchedulerInterface *child)
	{
		{
			std::lock_guard<SpinLock> guard(_globalLock);
			for (auto it = _idleChildren.begin(); it != _idleChildren.end(); ++it) {
				if (*it == child) {
					_idleChildren.erase(it);
					break;
				}
			}
		}
		
		if (_parent != nullptr) {
			_parent->unidleChild(this);
		}
	}
	
	inline void updateQueueThreshold(size_t queueThreshold)
	{
		if (queueThreshold < _queueThreshold) {
			_rebalance = true;
		}
		
		_queueThreshold = queueThreshold;
		
		for (TreeSchedulerInterface *child : _children) {
			child->updateQueueThreshold(_queueThreshold);
		}
	}
	
	inline void disableChild(__attribute__((unused)) TreeSchedulerInterface *child)
	{
		assert(_enabledChildren > 0);
		
		if ((--_enabledChildren) == 0 && _parent != nullptr) {
			std::vector<Task *> taskBatch = _queue->getTaskBatch(-1);
		
			if (taskBatch.size() > 0) {
				_parent->addTaskBatch(this, taskBatch, true);
			}
			_parent->disableChild(this);
		}
	}
	
	inline void enableChild(__attribute__((unused)) TreeSchedulerInterface *child)
	{
		++_enabledChildren;
	}
};

#endif // NODE_SCHEDULER_HPP
