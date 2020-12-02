/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "OpenAccAccelerator.hpp"

#include "hardware/places/ComputePlace.hpp"
#include "hardware/places/MemoryPlace.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/BlockingAPI.hpp"


ConfigVariable<bool> OpenAccAccelerator::_pinnedPolling("devices.openacc.polling.pinned", true);
ConfigVariable<size_t> OpenAccAccelerator::_usPollingPeriod("devices.openacc.polling.period_us", 500);


void OpenAccAccelerator::acceleratorServiceLoop()
{
	const size_t sleepTime = _usPollingPeriod.getValue();

	while (!shouldStopService()) {
		bool activeDevice = false;
		do {
			// Launch as many ready device tasks as possible
			while (isQueueAvailable()) {
				Task *task = Scheduler::getReadyTask(_computePlace);
				if (task == nullptr)
					break;

				runTask(task);
			}

			// Only set the active device if there have been tasks launched
			// Setting the device during e.g. bootstrap caused issues
			if (!_activeQueues.empty()) {
				if (!activeDevice) {
					activeDevice = true;
					setActiveDevice();
				}

				// Process the active events
				processQueues();
			}

			// Iterate while there are running tasks and pinned polling is enabled
		} while (_pinnedPolling && !_activeQueues.empty());

		// Sleep for 500 microseconds
		BlockingAPI::waitForUs(sleepTime);
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

