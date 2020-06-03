/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "OpenAccFunctions.hpp"
#include "OpenAccQueuePool.hpp"

OpenAccQueuePool::OpenAccQueuePool() :
	_maxAsyncQueues(OpenAccFunctions::getMaxQueues())
{
	// preallocate the default num of queues; on request expand up to _maxAsyncQueues
	size_t numQueues = OpenAccFunctions::getInitialQueueNum();
	size_t i;
	for (i = 1; i <= numQueues; i++) {	// count from 1; keep 0 for special cases (eg. in_final)
		OpenAccQueue *queue = new OpenAccQueue((int)i);
		assert(queue != nullptr);
		_queuePool.push_back(queue);
	}
	_nextAsyncId = i;
}

OpenAccQueuePool::~OpenAccQueuePool()
{
	while (! _queuePool.empty()) {
		OpenAccQueue *queue = _queuePool.front();
		_queuePool.pop_front();
		delete queue;
	}
}

OpenAccQueue *OpenAccQueuePool::getAsyncQueue()
{
	if (_queuePool.empty()) {
		return new OpenAccQueue(_nextAsyncId++);
	} else {
		OpenAccQueue *queue = _queuePool.front();
		_queuePool.pop_front();
		return queue;
	}
}

