/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2022 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <map>
#include <mutex>
#include <string>
#include <utility>

#include <nanos6.h>
#include <nanos6/library-mode.h>

#include "AddTask.hpp"
#include "SpawnFunction.hpp"
#include "lowlevel/SpinLock.hpp"
#include "monitoring/Monitoring.hpp"
#include "system/TrackingPoints.hpp"
#include "tasks/StreamManager.hpp"
#include "tasks/Task.hpp"

#include <InstrumentAddTask.hpp>


//! Static members
std::atomic<unsigned int> SpawnFunction::_pendingSpawnedFunctions(0);
std::map<SpawnFunction::task_info_key_t, nanos6_task_info_t> SpawnFunction::_spawnedFunctionInfos;
SpinLock SpawnFunction::_spawnedFunctionInfosLock;
nanos6_task_invocation_info_t SpawnFunction::_spawnedFunctionInvocationInfo = { "Spawned from external code" };


//! Args block of spawned functions
struct SpawnedFunctionArgsBlock {
	SpawnFunction::function_t _function;
	void *_args;
	SpawnFunction::function_t _completionCallback;
	void *_completionArgs;

	SpawnedFunctionArgsBlock() :
		_function(nullptr),
		_args(nullptr),
		_completionCallback(nullptr),
		_completionArgs(nullptr)
	{
	}
};


void SpawnFunction::spawnFunction(
	function_t function,
	void *args,
	function_t completionCallback,
	void *completionArgs,
	char const *label,
	bool fromUserCode
) {
	WorkerThread *workerThread = WorkerThread::getCurrentWorkerThread();
	Task *creator = nullptr;
	if (workerThread != nullptr) {
		creator = workerThread->getTask();
	}

	// Runtime Tracking Point - Entering the creation of a task
	TrackingPoints::enterSpawnFunction(creator, fromUserCode);

	// Increase the number of spawned functions in case it is
	// spawned from outside the runtime system
	if (fromUserCode) {
		_pendingSpawnedFunctions++;
	}

	nanos6_task_info_t *taskInfo = nullptr;
	{
		task_info_key_t taskInfoKey(function, (label != nullptr ? label : ""));

		std::lock_guard<SpinLock> guard(_spawnedFunctionInfosLock);
		auto itAndBool = _spawnedFunctionInfos.emplace(
			std::make_pair(taskInfoKey, nanos6_task_info_t())
		);
		auto it = itAndBool.first;
		taskInfo = &(it->second);

		// Check whether it is a new task info
		if (itAndBool.second) {
			// Make sure all task info's fields are initialized to zero
			std::memset(taskInfo, 0, sizeof(nanos6_task_info_t));

			// Allocate memory for the task impl info
			taskInfo->implementations = (nanos6_task_implementation_info_t *)
				calloc(1, sizeof(nanos6_task_implementation_info_t));
			assert(taskInfo->implementations != nullptr);

			taskInfo->implementation_count = 1;
			taskInfo->implementations[0].run = SpawnFunction::spawnedFunctionWrapper;
			taskInfo->implementations[0].device_type_id = nanos6_device_t::nanos6_host_device;

			// Use a copy since we do not know the actual lifetime of label
			taskInfo->implementations[0].task_type_label = it->first.second.c_str();
			taskInfo->implementations[0].declaration_source = "Spawned Task";

			// The completion callback will be called when the task is destroyed
			taskInfo->destroy_args_block = SpawnFunction::spawnedFunctionDestructor;
			
			// Since it is a new taskinfo, register it in the Instrumentation
			Instrument::registeredNewSpawnedTaskType(taskInfo);
			
			// If a taskinfo is created and it is new, we notify Monitoring so
			// a new type is created. If the taskinfo is not new, it will exist
			// and it is being used, so we don't need to call registerTasktype
			Monitoring::registerTasktype(taskInfo);
		}
	}

	// Create the task representing the spawned function
	Task *task = AddTask::createTask(
		taskInfo, &_spawnedFunctionInvocationInfo,
		nullptr, sizeof(SpawnedFunctionArgsBlock),
		nanos6_waiting_task
	);
	assert(task != nullptr);

	SpawnedFunctionArgsBlock *argsBlock =
		(SpawnedFunctionArgsBlock *) task->getArgsBlock();
	assert(argsBlock != nullptr);

	argsBlock->_function = function;
	argsBlock->_args = args;
	argsBlock->_completionCallback = completionCallback;
	argsBlock->_completionArgs = completionArgs;

	task->setSpawned();
#ifdef EXTRAE_ENABLED
	if (label != nullptr && strcmp(label, "main") == 0) {
		task->markAsMainTask();
	}
#endif

	// Submit the task without parent
	AddTask::submitTask(task, nullptr);

	// Runtime Tracking Point - Exiting the creation of a task
	TrackingPoints::exitSpawnFunction(creator, fromUserCode);
}

void SpawnFunction::spawnedFunctionWrapper(void *args, void *, nanos6_address_translation_entry_t *)
{
	SpawnedFunctionArgsBlock *argsBlock = (SpawnedFunctionArgsBlock *) args;
	assert(argsBlock != nullptr);

	// Call the user spawned function
	argsBlock->_function(argsBlock->_args);
}

void SpawnFunction::spawnedFunctionDestructor(void *args)
{
	SpawnedFunctionArgsBlock *argsBlock = (SpawnedFunctionArgsBlock *) args;
	assert(argsBlock != nullptr);

	// Call the user completion callback if present
	if (argsBlock->_completionCallback != nullptr) {
		argsBlock->_completionCallback(argsBlock->_completionArgs);
	}
}


//! Public API function to spawn functions
void nanos6_spawn_function(
	void (*function)(void *),
	void *args,
	void (*completion_callback)(void *),
	void *completion_args,
	char const *label
) {
	SpawnFunction::spawnFunction(function, args, completion_callback, completion_args, label, true);
}

//! Public API function to spawn functions in streams
void nanos6_stream_spawn_function(
	void (*function)(void *),
	void *args,
	void (*callback)(void *),
	void *callback_args,
	char const *label,
	size_t stream_id
) {
	StreamManager::createFunction(function, args, callback, callback_args, label, stream_id);
}
