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
	// Workaround to avoid the overhead of repetitive/redundant setActiveDevice() calls
	bool activeDevice = false;

	do {
		if (isQueueAvailable()) {
			task = Scheduler::getReadyTask(_computePlace);
			if (task != nullptr) {
				runTask(task);
			}
		}
		// Check if there is merit in setting the device;
		// Only do the setActiveDevice if there have been tasks launched
		// If we set it before any OpenACC work has run (e.g. during bootstrap) it causes
		// erroneous behavior. If we don't set it here, the completion checks may not occur correctly
		if (!activeDevice && !_activeQueues.empty()) {
			setActiveDevice();
			activeDevice = true;
		}
		processQueues();
	} while (!_activeQueues.empty() && worker);

	// If process was run by LeaderThread, request a WorkerThread to continue
	if (!worker && (task != nullptr || !_activeQueues.empty())) {
		CPUManager::executeCPUManagerPolicy(nullptr, REQUEST_CPUS, 1);
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

