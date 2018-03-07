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
	EnvironmentVariable<size_t> _minQueueThreshold;
	EnvironmentVariable<size_t> _maxQueueThreshold;
	EnvironmentVariable<size_t> _pollingIterations;
	
	std::deque<SchedulerInterface *> _idleChildren;
	SchedulerQueueInterface *_queue;
	
	NodeScheduler *_parent;
	std::vector<SchedulerInterface *> _children;
	
	SpinLock _idleChildrenLock;
	
	inline void handleQueueOverflow()
	{
		if (_parent != nullptr) {
			std::vector<Task *> taskBatch = _queue->getTaskBatch(_maxQueueThreshold - _minQueueThreshold);
			_parent->addTaskBatch(taskBatch);
		} else {
			// Do nothing, just stack more tasks
		}
	}

public:
	NodeScheduler(NodeScheduler *parent = nullptr) :
		_minQueueThreshold("NANOS6_SCHEDULER_QUEUE_MIN_THRESHOLD", 10),
		_maxQueueThreshold("NANOS6_SCHEDULER_QUEUE_MAX_THRESHOLD", 20),
		_pollingIterations("NANOS6_SCHEDULER_POLLING_ITER", 100000),
		_parent(parent)
	{
		assert(_maxQueueThreshold >= _minQueueThreshold);
		
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

	inline void addTaskBatch(std::vector<Task *> &taskBatch)
	{
		SchedulerInterface *idleChild = nullptr;
		
		{
			std::lock_guard<SpinLock> guard(_idleChildrenLock);
			if (_idleChildren.size() > 0) {
				idleChild = _idleChildren.front();
				_idleChildren.pop_front();
			}
		}
		
		if (idleChild != nullptr) {
			idleChild->addTaskBatch(taskBatch);
		} else {
			size_t elements = _queue->addTaskBatch(taskBatch);
			if (elements > _maxQueueThreshold) {
				handleQueueOverflow();
			}
		}
	}
	
	inline void getTask(SchedulerInterface *child)
	{
		std::vector<Task *> taskBatch = _queue->getTaskBatch(_minQueueThreshold);
		
		if (taskBatch.size() > 0) {
			child->addTaskBatch(taskBatch);
		} else {
			{
				std::lock_guard<SpinLock> guard(_idleChildrenLock);
				_idleChildren.push_back(child);
			}
			
			if (_parent != nullptr) {
				_parent->getTask(this);
			}
		}
	}
	
	inline void setChild(SchedulerInterface *child)
	{
		_children.push_back(child);
	}
	
	inline void unidleChild(SchedulerInterface *child)
	{
		{
			std::lock_guard<SpinLock> guard(_idleChildrenLock);
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
};

#endif // NODE_SCHEDULER_HPP
