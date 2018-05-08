/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef NODE_SCHEDULER_HPP
#define NODE_SCHEDULER_HPP

#include <vector>

#include "executors/threads/ThreadManager.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "SchedulerInterface.hpp"
#include "SchedulerQueueInterface.hpp"

class NodeScheduler: public SchedulerInterface {
private:
	EnvironmentVariable<size_t> _pollingIterations;
	
	std::atomic<size_t> _queueThreshold;
	
	std::deque<SchedulerInterface *> _idleChildren;
	SchedulerQueueInterface *_queue;
	
	NodeScheduler *_parent;
	std::vector<SchedulerInterface *> _children;
	
	SpinLock _globalLock;
	SpinLock _thresholdLock;
	
	inline void handleQueueOverflow()
	{
		if (_parent != nullptr) {
			size_t elements = _queueThreshold / 2;
			
			if (elements == 0) {
				elements = 1;
			}
			
			std::vector<Task *> taskBatch = _queue->getTaskBatch(elements);
			if (taskBatch.size() > 0) {
				// queue might have been emptied just a moment ago
				_parent->addTaskBatch(this, taskBatch);
			}
		} else {
			// Increase threshold and propagate downwards
			std::lock_guard<SpinLock> guard(_thresholdLock);
			size_t th = _queueThreshold * 2;
			
			if (th == 0) {
				th = 2;
			}
			
			updateQueueThreshold(th);
		}
	}

public:
	NodeScheduler(NodeScheduler *parent = nullptr) :
		_pollingIterations("NANOS6_SCHEDULER_POLLING_ITER", 100000),
		_queueThreshold(0),
		_parent(parent)
	{
		_queue = SchedulerQueueInterface::initialize();
		if (_parent != nullptr) {
			_parent->setChild(this);
		}
	}
	
	~NodeScheduler()
	{
		delete _queue;
		for (SchedulerInterface *sched : _children) {
			delete sched;
		}
	}

	inline void addTaskBatch(SchedulerInterface *who, std::vector<Task *> &taskBatch)
	{
		SchedulerInterface *idleChild = who;
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
			idleChild->addTaskBatch(this, taskBatch);
		} else if (overflow) {
			handleQueueOverflow();
		}
	}
	
	inline void getTask(SchedulerInterface *child)
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
			}
		}
		
		// Outside lock, call other nodes
		if (taskBatch.size() > 0) {
			child->addTaskBatch(this, taskBatch);
		} else {
			if (_parent != nullptr) {
				_parent->getTask(this);
			}
		}
		
		if (_parent == nullptr && _queueThreshold > 0) {
			// Reduce threshold and propagate
			std::lock_guard<SpinLock> guard(_thresholdLock);
			updateQueueThreshold(_queueThreshold / 2);
		}
	}
	
	inline void setChild(SchedulerInterface *child)
	{
		_children.push_back(child);
	}
	
	inline void unidleChild(SchedulerInterface *child)
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
		_queueThreshold = queueThreshold;
		
		for (SchedulerInterface *child : _children) {
			child->updateQueueThreshold(_queueThreshold);
		}
	}
};

#endif // NODE_SCHEDULER_HPP
