/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6.h>

#include <MemoryAllocator.hpp>

#include "DataAccessRegistration.hpp"
#include "Throttle.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/ompss/TaskBlocking.hpp"
#include "system/ompss/TaskWait.hpp"
#include "tasks/Task.hpp"

int Throttle::_pressure;
ConfigVariable<bool> Throttle::_enabled("throttle.enabled");
ConfigVariable<int> Throttle::_throttleTasks("throttle.tasks");
ConfigVariable<int> Throttle::_throttlePressure("throttle.pressure");
ConfigVariable<StringifiedMemorySize> Throttle::_throttleMem("throttle.max_memory");
ConfigVariable<size_t> Throttle::_throttlePollingPeriod("throttle.polling_period_us");

std::atomic<bool> Throttle::_stopService;
std::atomic<bool> Throttle::_finishedService;

void Throttle::initialize()
{
	// If we don't have usage statistics, we cannot enable the Throttle
	if (!MemoryAllocator::hasUsageStatistics())
		_enabled.setValue(false);

	if (!_enabled)
		return;

	// The default max memory is half of the hosts physical memory
	if (_throttleMem.getValue() == 0)
		_throttleMem.setValue(HardwareInfo::getPhysicalMemorySize() / 2);

	_pressure = 0;
	_stopService = false;
	_finishedService = false;

	// Sanity check for the histeresis values
	FatalErrorHandler::failIf((_throttleTasks < 0), "Throttle tasks must be > 0");
	FatalErrorHandler::failIf((_throttlePressure > 100 || _throttlePressure < 0), "Throttle pressure trigger has to be between 0 and 100%");

	// Spawn service function
	SpawnFunction::spawnFunction(
		evaluate, nullptr,
		complete, nullptr,
		"Throttle evaluate", false
	);
}

void Throttle::shutdown()
{
	if (_enabled.getValue()) {
		_stopService = true;

		// Wait until service is finished
		while (!_finishedService.load(std::memory_order_relaxed));
	}
}

void Throttle::evaluate(void *)
{
	const size_t sleepTime = _throttlePollingPeriod.getValue();
	assert(_enabled);
	assert(_throttleMem.getValue() != 0);

	while (!_stopService.load(std::memory_order_relaxed)) {
		size_t memoryUsage = MemoryAllocator::getMemoryUsage();
		_pressure = std::min((memoryUsage * 100) / _throttleMem.getValue(), (size_t)100);

		// Sleep for a configured amount of microseconds
		BlockingAPI::waitForUs(sleepTime);
	}
}

void Throttle::complete(void *)
{
	assert(_stopService);

	_finishedService = true;
}

// Each task has a maximum number of child tasks, which decreases at a 10x rate per nesting level
// determined by throttle.tasks. Also, when the memory pressure reaches throttle.max_memory
// the number of tasks dicreases linearly between that point and 100% memory pressure. At a 100%
// memory pressure, there is only 1 allowed tasks, which will trigger a taskwait to reduce
// memory usage.
int Throttle::getAllowedTasks(int nestingLevel)
{
	int standardAllowedTasks = _throttleTasks;
	int startDecay = _throttlePressure;

	standardAllowedTasks /= std::max(nestingLevel * 10, 1);

	assert(_pressure >= 0 && _pressure <= 100);

	if (_pressure < startDecay) {
		// How many alive tasks do we allow x nesting level?
		assert(_pressure != 100);
		return standardAllowedTasks;
	} else {
		if (_pressure == 100)
			return 1;

		int decay = _pressure - startDecay;
		// normalize the decay over 100%
		decay = (decay * 100) / (100 - startDecay);
		// calculate tasks we have to remove
		decay = std::min((standardAllowedTasks * decay) / 100, standardAllowedTasks - 1);

		return standardAllowedTasks - decay;
	}
}

bool Throttle::engage(Task *creator, WorkerThread *workerThread)
{
	assert(creator != nullptr);
	assert(workerThread != nullptr);
	assert(workerThread->getTask() == creator);

	// We cannot safely throttle taskloops because we cannot prevent
	// a taskloop from creating child tasks.
	if (creator->isTaskloop())
		return false;

	// How many child tasks is this creator allowed?
	int nestingLevel = creator->getNestingLevel();
	int allowedChildTasks = getAllowedTasks(nestingLevel);

	// No need to activate if very few child tasks exist
	if (creator->getPendingChildTasks() <= allowedChildTasks)
		return false;

	CPU *currentCPU = workerThread->getComputePlace();
	assert(currentCPU != nullptr);

	// Let's try and give the worker thread a different task to execute while we wait
	Task *replacement = nullptr;

	if (allowedChildTasks != 1)
		replacement = Scheduler::getReadyTask(currentCPU);

	if (replacement != nullptr && workerThread->isTaskReplaceable()) {
		workerThread->replaceTask(replacement);
		workerThread->handleTask(currentCPU);

		// Restore
		workerThread->restoreTask(creator);
		return true;
	} else {
		// There is nothing else to do. Let's run a taskwait then
		TaskWait::taskWait("Throttle");
		return false;
	}
}
