/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "OpenAccAccelerator.hpp"

#include "hardware/places/ComputePlace.hpp"
#include "hardware/places/MemoryPlace.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/BlockingAPI.hpp"


void OpenAccAccelerator::acceleratorServiceLoop()
{
	while (!shouldStopService()) {
		while (isQueueAvailable()) {
			Task *task = Scheduler::getReadyTask(_computePlace);
			if (task == nullptr)
				break;

			runTask(task);
		}

		// Only do the setActiveDevice if there have been tasks launched
		// Having setActiveDevice calls during e.g. bootstrap caused issues
		if (!_activeQueues.empty()) {
			setActiveDevice();
			processQueues();
		}

		// Sleep for 500 microseconds
		BlockingAPI::waitForUs(500);
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

