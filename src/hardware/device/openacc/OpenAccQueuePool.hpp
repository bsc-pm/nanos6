/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef OPENACC_QUEUE_POOL_HPP
#define OPENACC_QUEUE_POOL_HPP

#include "OpenAccFunctions.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "tasks/Task.hpp"

class OpenAccQueue {
private:
	int _queueId;
	Task *_task;

public:
	OpenAccQueue(int id) :
		_queueId(id),
		_task(nullptr)
	{
	}

	~OpenAccQueue()
	{
	}

	inline int getQueueId() const
	{
		return _queueId;
	}

	inline bool isFinished() const
	{
		return OpenAccFunctions::asyncFinished(_queueId);
	}

	inline Task *getTask() const
	{
		return _task;
	}

	inline void setTask(Task *task)
	{
		_task = task;
	}

};

class OpenAccQueuePool {
private:
	std::deque<OpenAccQueue *> _queuePool;

	// The id that the next newly created (if pool is empty) queue will have
	size_t _nextAsyncId;

	// Defined in NANOS6_OPENACC_MAX_QUEUES, maximum number we can totally have
	size_t _maxAsyncQueues;

public:
	OpenAccQueuePool() :
		_maxAsyncQueues(OpenAccFunctions::getMaxQueues())
	{
		// preallocate the default num of queues; on request expand up to _maxAsyncQueues
		size_t numQueues = OpenAccFunctions::getInitialQueueNum();
		FatalErrorHandler::failIf(numQueues > _maxAsyncQueues,
			"NANOS6_OPENACC_DEFAULT_QUEUES can't be greater than NANOS6_OPENACC_MAX_QUEUES",
			"\nPlease set environment variables accordingly");

		for (size_t i = 1; i <= numQueues; i++) {	// count from 1; keep 0 for special cases (eg. in_final)
			OpenAccQueue *queue = new OpenAccQueue((int)i);
			assert(queue != nullptr);
			_queuePool.push_back(queue);
		}
		_nextAsyncId = numQueues + 1;
	}

	~OpenAccQueuePool()
	{
		while (!_queuePool.empty()) {
			OpenAccQueue *queue = _queuePool.front();
			_queuePool.pop_front();
			delete queue;
		}
	}

	// We have available queues in 2 cases:
	//   1. The existing queues pool is not empty
	//   2. We haven't reached the defined MAX queue number, so we can allocate more
	//      (see NANOS6_OPENACC_MAX_QUEUES)
	inline bool isQueueAvailable() const
	{
		if (_queuePool.empty()) {
			if (_nextAsyncId > _maxAsyncQueues) {
				return false;
			}
		}
		return true;
	}

	inline OpenAccQueue *getAsyncQueue()
	{
		if (_queuePool.empty()) {
			assert(_nextAsyncId <= _maxAsyncQueues);
			return new OpenAccQueue(_nextAsyncId++);
		} else {
			OpenAccQueue *queue = _queuePool.front();
			_queuePool.pop_front();
			return queue;
		}
	}

	inline void releaseAsyncQueue(OpenAccQueue *queue)
	{
		_queuePool.push_back(queue);
	}

};

#endif // OPENACC_QUEUE_POOL_HPP
