/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef OPENACC_QUEUE_HPP
#define OPENACC_QUEUE_HPP

#include "tasks/Task.hpp"

class OpenAccQueue {
private:
	int _queueId;
	Task *_task;

public:
	OpenAccQueue(int id) :
		_queueId(id)
	{
	}

	~OpenAccQueue()
	{
	}

	inline int getQueueId()
	{
		return _queueId;
	}

	inline bool isFinished()
	{
		return OpenAccFunctions::asyncFinished(_queueId);
	}

	inline Task *getTask()
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
	OpenAccQueuePool();

	~OpenAccQueuePool();

	// We have available queues in 2 cases:
	//   1. The existing queues pool is not empty
	//   2. We haven't reached the defined MAX queue number, so we can allocate more
	//      (see NANOS6_OPENACC_MAX_QUEUES)
	inline bool isQueueAvailable()
	{
		if (_queuePool.empty()) {
			if (_nextAsyncId > _maxAsyncQueues) {
				return false;
			}
		}
		return true;
	}

	OpenAccQueue *getAsyncQueue();

	inline void releaseAsyncQueue(OpenAccQueue *queue)
	{
		_queuePool.push_back(queue);
	}

};

#endif // OPENACC_QUEUE_HPP
