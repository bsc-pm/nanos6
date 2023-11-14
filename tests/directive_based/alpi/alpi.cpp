/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#include "Atomic.hpp"
#include "TestAnyProtocolProducer.hpp"

#include <nanos6/alpi.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(f...)                                                                \
do {                                                                               \
	const int __r = f;                                                             \
	if (__r) {                                                                     \
		fprintf(stderr, "Error: '%s' [%s:%i]: %i\n", #f, __FILE__, __LINE__, __r); \
		exit(EXIT_FAILURE);                                                        \
	}                                                                              \
} while (0)

TestAnyProtocolProducer tap;

Atomic<struct alpi_task *> blockedTask(NULL);
Atomic<struct alpi_task *> eventsTask(NULL);

// Get current task and check whether it has a handle
struct alpi_task *getCurrentTask()
{
	struct alpi_task *task;
	CHECK(alpi_task_self(&task));

	tap.evaluate(task != NULL, "The current task has a valid task handle");

	return task;
}

// Perform several checks regarding the versions
void performVersionChecks()
{
	int required[2], provided[2], err;

	CHECK(alpi_version_get(&provided[0], &provided[1]));
	tap.evaluate(provided[0] == ALPI_VERSION_MAJOR, "Same provided and requested major versions");
	tap.evaluate(provided[1] == ALPI_VERSION_MINOR, "Same provided and requested minor versions");

	required[0] = ALPI_VERSION_MAJOR;
	required[1] = 0;
	err = alpi_version_check(required[0], required[1]);
	tap.evaluate(!err, "Requesting same or lower minor version is valid");

	required[0] = ALPI_VERSION_MAJOR;
	required[1] = ALPI_VERSION_MINOR + 1;
	err = alpi_version_check(required[0], required[1]);
	tap.evaluate(err == ALPI_ERR_VERSION, "Requesting higher minor version is invalid");

	required[0] = ALPI_VERSION_MAJOR - 1;
	required[1] = 0;
	err = alpi_version_check(required[0], required[1]);
	tap.evaluate(err == ALPI_ERR_VERSION, "Requesting lower major version is invalid");

	required[0] = ALPI_VERSION_MAJOR + 1;
	required[1] = 0;
	err = alpi_version_check(required[0], required[1]);
	tap.evaluate(err == ALPI_ERR_VERSION, "Requesting higher major version is invalid");
}

// Perform several checks regarding the error handling
void performErrorChecks()
{
	const char *string;
	string = alpi_error_string(ALPI_SUCCESS);
	tap.evaluate(string && std::strlen(string) > 0, "String of ALPI_SUCCESS is valid");

	string = alpi_error_string(ALPI_ERR_PARAMETER);
	tap.evaluate(string && std::strlen(string) > 0, "String of ALPI_ERR_PARAMETER is valid");

	string = alpi_error_string(-1);
	tap.evaluate(string && std::strlen(string) > 0, "String of an unknown error is valid");

	string = alpi_error_string(ALPI_ERR_MAX);
	tap.evaluate(string && std::strlen(string) > 0, "String of an unknown error is valid");
}

// Perform several checks regarding the CPUs
void performCPUChecks()
{
	uint64_t count, logical, system;
	CHECK(alpi_cpu_count(&count));
	CHECK(alpi_cpu_logical_id(&logical));
	CHECK(alpi_cpu_system_id(&system));

	tap.evaluate(count > 0, "There is at least one CPU available");
	tap.evaluate(logical >= 0 && logical < count, "The logical CPU id is valid");
}

// The body of the polling task
void taskPollingBody(void *args)
{
	tap.evaluate(args != NULL, "Spawned task body received correct args");

	performCPUChecks();

	// Wait until the blocked and the events tasks have done their actions
	do {
		uint64_t slept;
		CHECK(alpi_task_waitfor_ns(10000, &slept));
	} while (!blockedTask.load() || !eventsTask.load());

	// Retrieve all the task handles
	struct alpi_task * toUnblock = blockedTask.load();
	struct alpi_task * toDecrease = eventsTask.load();
	struct alpi_task * pollingTask = getCurrentTask();

	tap.evaluate(toUnblock != toDecrease, "Different tasks have different handle");
	tap.evaluate(toUnblock != pollingTask, "Different tasks have different handle");
	tap.evaluate(toDecrease != pollingTask, "Different tasks have different handle");

	// Mark those tasks processed by nullifying the handles
	blockedTask.store(NULL);
	eventsTask.store(NULL);

	// Unblock and decrease the events of the tasks
	CHECK(alpi_task_unblock(toUnblock));
	CHECK(alpi_task_events_decrease(toDecrease, 1));
}

// The completion callback of the polling task
void taskPollingCompleted(void *args)
{
	tap.evaluate(args, "Spawned task completion received correct args");

	// Notify the main function that the spawned task has completed
	Atomic<bool> *completed = static_cast<Atomic<bool> *>(args);
	completed->store(true);
}

// The body of the task that blocks
void taskBlocksBody()
{
	performCPUChecks();

	// Notify the polling task and block the current task
	struct alpi_task *task = getCurrentTask();
	blockedTask.store(task);
	CHECK(alpi_task_block(task));

	// After being resumed, check that the polling task actually processed the
	// unblocking of the task
	tap.evaluate(!blockedTask.load(), "Blocked task was actually processed");
}

// The body of the task that registers events
void taskEventsBody()
{
	performCPUChecks();

	// Increase the task's events and notify the polling task
	struct alpi_task *task = getCurrentTask();
	CHECK(alpi_task_events_increase(task, 1));
	eventsTask.store(task);
}

int main()
{
	tap.registerNewTests(30);
	tap.begin();

	// Check that the main function is executed by a task
	getCurrentTask();

	// Perform checks about versions, error handling and the available CPUs
	performVersionChecks();
	performErrorChecks();
	performCPUChecks();

	// Instantiate two tasks that will block and register events, respectively
	#pragma oss task label("blocked task")
	taskBlocksBody();

	#pragma oss task label("events task")
	taskEventsBody();

	uint64_t attributesSize;
	struct alpi_attr *attributes;

	// Create an empty task attribute structure. For the moment, there is no
	// valid attribute so the runtime just creates the default attribute
	CHECK(alpi_attr_create(&attributes));
	CHECK(alpi_attr_init(attributes));
	CHECK(alpi_attr_size(&attributesSize));

	tap.evaluate(attributes == NULL, "The attributes are the default");
	tap.evaluate(attributesSize == 0, "The attributes have zero size");

	// Spawn a task to unblock and fulfill the events of the previous tasks
	Atomic<bool> completed(false);
	CHECK(alpi_task_spawn(
		taskPollingBody, static_cast<void *>(&attributesSize),
		taskPollingCompleted, static_cast<void *>(&completed),
		"polling", attributes));

	// Now it is safe to destroy the attributes
	CHECK(alpi_attr_destroy(attributes));

	// Wait until the spawned task has finished
	do {
		CHECK(alpi_task_waitfor_ns(500000, NULL));
	} while (!completed);

	#pragma oss taskwait

	tap.end();

	return 0;
}
