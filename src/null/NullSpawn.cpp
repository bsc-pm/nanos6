/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6.h>
#include <nanos6/library-mode.h>

#include <cassert>

class Task;

static nanos6_task_invocation_info _spawnedFunctionInvocationInfo = { "Spawned from external code" };

struct SpawnedFunctionArgsBlock {
	void (*_function)(void *);
	void *_args;
	void (*_completion_callback)(void *);
	void *_completion_args;
	
	SpawnedFunctionArgsBlock()
	: _function(nullptr), _args(nullptr), _completion_callback(nullptr), _completion_args(nullptr)
	{
	}
};


static void nanos6_spawned_function_wrapper(void *args, __attribute__((unused)) nanos6_taskloop_bounds_t *bounds)
{
	SpawnedFunctionArgsBlock *argsBlock = (SpawnedFunctionArgsBlock *) args;
	assert(argsBlock != nullptr);
	
	argsBlock->_function(argsBlock->_args);
	
	if (argsBlock->_completion_callback != nullptr) {
		argsBlock->_completion_callback(argsBlock->_completion_args);
	}
}


void nanos6_spawn_function(void (*function)(void *), void *args, void (*completion_callback)(void *), void *completion_args, char const *label)
{
	nanos6_task_info taskInfo;
	taskInfo.run = nanos6_spawned_function_wrapper;
	taskInfo.register_depinfo = nullptr;
	taskInfo.task_label = label;
	taskInfo.declaration_source = "Spawned Task";
	taskInfo.get_cost = nullptr;
	
	SpawnedFunctionArgsBlock *argsBlock = nullptr;
	nanos6_taskloop_bounds_t *bounds = nullptr;
	Task *task = nullptr;
	nanos6_create_task(&taskInfo, &_spawnedFunctionInvocationInfo, sizeof(SpawnedFunctionArgsBlock), (void **) &argsBlock, (void **) &bounds, (void **) &task, 0);
	
	assert(argsBlock != nullptr);
	assert(task != nullptr);
	
	argsBlock->_function = function;
	argsBlock->_args = args;
	argsBlock->_completion_callback = completion_callback;
	argsBlock->_completion_args = completion_args;
	
	nanos6_submit_task(task);
}

