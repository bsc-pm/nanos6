/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "DataAccessRegistration.hpp"
#include "Throttle.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "hardware-counters/HardwareCounters.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/ompss/TaskBlocking.hpp"
#include "tasks/Task.hpp"

#include <nanos6.h>

#include <InstrumentTaskStatus.hpp>
#include <InstrumentTaskWait.hpp>
#include <MemoryAllocator.hpp>
#include <Monitoring.hpp>

#include <random>
#include <iostream>

int Throttle::_pressure;
EnvironmentVariable<bool> Throttle::_enabled("NANOS6_THROTTLE_ENABLE", false);
EnvironmentVariable<int> Throttle::_throttleTasks("NANOS6_THROTTLE_TASKS", 5000000);
EnvironmentVariable<int> Throttle::_throttlePressure("NANOS6_THROTTLE_PRESSURE", 70);
EnvironmentVariable<StringifiedMemorySize> Throttle::_throttleMem("NANOS6_THROTTLE_MAX_MEMORY", ((size_t) 0));

void Throttle::initialize()
{
	// If we don't have usage statistics, we cannot enable the Throttle
	if (!MemoryAllocator::hasUsageStatistics())
		_enabled.setValue(false);

	// The default max memory is half of the hosts physical memory.
	if (_throttleMem.getValue() == 0)
		_throttleMem.setValue(HardwareInfo::getPhysicalMemorySize() / 2);

	_pressure = 0;

	// Sanity check for the histeresis values
	FatalErrorHandler::failIf((_throttleTasks < 0), "Throttle tasks must be > 0");
	FatalErrorHandler::failIf((_throttlePressure > 100 || _throttlePressure < 0), "Throttle pressure trigger has to be between 0 and 100%");

	if (_enabled.getValue()) {
		nanos6_register_polling_service(
			"Throttle Evaluation",
			evaluate,
			nullptr
		);
	}
}

void Throttle::shutdown()
{
	if (_enabled.getValue()) {
		nanos6_unregister_polling_service(
			"Throttle Evaluation",
			evaluate,
			nullptr
		);
	}
}

int Throttle::evaluate(void *)
{
	// As this is a constexpr, it should be optimized away at compile-time
	// if the runtime was configured without jemalloc.
	if (MemoryAllocator::hasUsageStatistics()) {
		if (_throttleMem.getValue() == 0) {
			_pressure = 0;
		} else {
			size_t memoryUsage = MemoryAllocator::getMemoryUsage();
			_pressure = std::min((memoryUsage * 100) / _throttleMem.getValue(), (size_t)100);
		}
	}

	return 0;
}

// Each task has a maximum number of child tasks, which decreases at a 10x rate per nesting level
// determined by NANOS6_THROTTLE_TASKS. Also, when the memory pressure reaches NANOS6_THROTTLE_MAX_MEMORY
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

bool Throttle::engage(Task *task, WorkerThread *workerThread)
{
	assert(task != nullptr);
	assert(workerThread != nullptr);
	assert(workerThread->getTask() == task);

	// We cannot safely throttle taskloops because we cannot prevent
	// a taskloop from creating child tasks.
	if (task->isTaskloop())
		return false;

	// How many child tasks is this task allowed?
	int nestingLevel = task->getNestingLevel();
	int allowedChildTasks = getAllowedTasks(nestingLevel);

	// No need to activate if very few child tasks exist
	if (task->pendingChildTasks() <= allowedChildTasks)
		return false;

	CPU *currentCPU = workerThread->getComputePlace();
	assert(currentCPU != nullptr);

	// Let's try and give the worker thread a different task to execute while we wait.
	Task *replacement = nullptr;

	if (allowedChildTasks != 1)
		replacement = Scheduler::getReadyTask(currentCPU);

	if (replacement != nullptr) {
		workerThread->unassignTask();
		workerThread->setTask(replacement);
		workerThread->handleTask(currentCPU);

		// Restore
		workerThread->setTask(task);
		return true;
	} else {
		// There is nothing else to do. Let's run a taskwait then.
		nanos6_taskwait("Throttle");
		return false;
	}
}
