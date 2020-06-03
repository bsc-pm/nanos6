/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "OpenAccAccelerator.hpp"

#include "hardware/places/ComputePlace.hpp"
#include "hardware/places/MemoryPlace.hpp"
#include "scheduling/Scheduler.hpp"

int OpenAccAccelerator::pollingService(void *data)
{
	OpenAccAccelerator *accel = (OpenAccAccelerator *)data;
	assert(accel != nullptr);

	accel->acceleratorServiceLoop();
	return 0;
}

void OpenAccAccelerator::acceleratorServiceLoop()
{
	// Check if the thread running the service is a WorkerThread. nullptr means LeaderThread
	bool worker = (WorkerThread::getCurrentWorkerThread() != nullptr);

	Task *task = nullptr;

	do {
		if (isQueueAvailable()) {
			task = Scheduler::getReadyTask(_computePlace);
			if (task != nullptr) {
				runTask(task);
			}
		}
		processQueues();
	} while (_activeQueues.size() != 0 && worker);

	// If process was run by LeaderThread, request a WorkerThread to continue
	if (!worker && (task != nullptr || !_activeQueues.empty())) {
		CPUManager::executeCPUManagerPolicy(nullptr, ADDED_TASKS, 1);
	}
}

void OpenAccAccelerator::processQueues()
{
	auto it = _activeQueues.begin();
	while (it != _activeQueues.end()) {
		OpenAccQueue *queue = *it;
		assert(queue != nullptr);
		if (queue->isFinished()) {
			finishTask(queue->getTask());
			it = _activeQueues.erase(it);
		} else {
			it++;
		}
	}
}

