/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023-2024 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include "nanos6/alpi.h"

#include "executors/threads/CPU.hpp"
#include "executors/threads/CPUManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "executors/threads/WorkerThreadImplementation.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "system/BlockingAPI.hpp"
#include "system/EventsAPI.hpp"
#include "system/SpawnFunction.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

extern "C" {

static constexpr const char *Errors[ALPI_ERR_MAX] = {
	[ALPI_SUCCESS] =             "Operation succeeded",
	[ALPI_ERR_VERSION] =         "Incompatible version",
	[ALPI_ERR_NOT_INITIALIZED] = "Runtime system not initialized",
	[ALPI_ERR_PARAMETER] =       "Invalid parameter",
	[ALPI_ERR_OUT_OF_MEMORY] =   "Failed to allocate memory",
	[ALPI_ERR_OUTSIDE_TASK] =    "Must run within a task",
	[ALPI_ERR_UNKNOWN] =         "Unknown error",
};

const char *alpi_error_string(int err)
{
	if (err < 0 || err >= ALPI_ERR_MAX)
		return "Error code not recognized";

	return Errors[err];
}

int alpi_version_check(int major, int minor)
{
	if (major != ALPI_VERSION_MAJOR)
		return ALPI_ERR_VERSION;
	if (minor > ALPI_VERSION_MINOR)
		return ALPI_ERR_VERSION;
	return 0;
}

int alpi_version_get(int *major, int *minor)
{
	if (major == nullptr || minor == nullptr)
		return ALPI_ERR_PARAMETER;

	*major = ALPI_VERSION_MAJOR;
	*minor = ALPI_VERSION_MINOR;
	return 0;
}

int alpi_task_self(struct alpi_task **handle)
{
	if (handle == nullptr)
		return ALPI_ERR_PARAMETER;

	Task *task = WorkerThread::getCurrentTask();
	*handle = reinterpret_cast<struct alpi_task *>(task);
	return 0;
}

int alpi_task_block(struct alpi_task *)
{
	BlockingAPI::blockCurrentTask(true);
	return 0;
}

int alpi_task_unblock(struct alpi_task *handle)
{
	Task *task = reinterpret_cast<Task *>(handle);
	if (task == nullptr)
		return ALPI_ERR_PARAMETER;

	BlockingAPI::unblockTask(task, true);
	return 0;
}

int alpi_task_events_increase(struct alpi_task *, uint64_t increment)
{
	if (increment == 0)
		return ALPI_ERR_PARAMETER;

	EventsAPI::increaseCurrentTaskEvents(increment);
	return 0;
}

int alpi_task_events_decrease(struct alpi_task *handle, uint64_t decrement)
{
	if (decrement == 0)
		return 0;

	Task *task = reinterpret_cast<Task *>(handle);
	if (task == nullptr)
		return ALPI_ERR_PARAMETER;

	EventsAPI::decreaseTaskEvents(task, decrement);
	return 0;
}

int alpi_task_events_test(struct alpi_task *, uint64_t *has_events)
{
	if (has_events == nullptr)
		return ALPI_ERR_PARAMETER;

	Task *task = WorkerThread::getCurrentTask();
	if (task == nullptr)
		return ALPI_ERR_OUTSIDE_TASK;

	int count = task->getReleaseCount();
	assert(count > 0);

	*has_events = count - 1;

	return 0;
}

int alpi_task_waitfor_ns(uint64_t target_ns, uint64_t *actual_ns)
{
	if (WorkerThread::getCurrentTask() == nullptr)
		return ALPI_ERR_OUTSIDE_TASK;

	uint64_t actual = BlockingAPI::waitForUs(target_ns / 1000);
	if (actual_ns)
		*actual_ns = actual * 1000;
	return 0;
}

int alpi_attr_create(struct alpi_attr **attr)
{
	if (attr == nullptr)
		return ALPI_ERR_PARAMETER;

	*attr = nullptr;
	return 0;
}

int alpi_attr_destroy(struct alpi_attr *)
{
	return 0;
}

int alpi_attr_init(struct alpi_attr *)
{
	return 0;
}

int alpi_attr_size(uint64_t *attr_size)
{
	if (attr_size == nullptr)
		return ALPI_ERR_PARAMETER;

	*attr_size = 0;
	return 0;
}

int alpi_task_spawn(
	void (*body)(void *),
	void *body_args,
	void (*completion_callback)(void *),
	void *completion_args,
	const char *label,
	const struct alpi_attr *)
{
	if (body == nullptr || completion_callback == nullptr)
		return ALPI_ERR_PARAMETER;

	SpawnFunction::spawnFunction(body, body_args,
		completion_callback, completion_args,
		label, true);
	return 0;
}

int alpi_task_suspend_mode_set(struct alpi_task *, alpi_suspend_mode_t, uint64_t)
{
	FatalErrorHandler::fail("Function ", __func__, " not implemented");
	return ALPI_ERR_UNKNOWN;
}

int alpi_task_suspend(struct alpi_task *)
{
	FatalErrorHandler::fail("Function ", __func__, " not implemented");
	return ALPI_ERR_UNKNOWN;
}

int alpi_cpu_count(uint64_t *count)
{
	if (count == nullptr)
		return ALPI_ERR_PARAMETER;

	*count = CPUManager::getTotalCPUs();
	return 0;
}

int alpi_cpu_logical_id(uint64_t *logical_id)
{
	if (logical_id == nullptr)
		return ALPI_ERR_PARAMETER;

	WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
	if (thread == nullptr)
		return ALPI_ERR_OUTSIDE_TASK;

	CPU *CPU = thread->getComputePlace();
	assert(CPU != nullptr);

	*logical_id = CPU->getIndex();
	return 0;
}

int alpi_cpu_system_id(uint64_t *system_id)
{
	if (system_id == nullptr)
		return ALPI_ERR_PARAMETER;

	WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
	if (thread == nullptr)
		return ALPI_ERR_OUTSIDE_TASK;

	CPU *CPU = thread->getComputePlace();
	assert(CPU != nullptr);

	*system_id = CPU->getSystemCPUId();
	return 0;
}

} // extern C
