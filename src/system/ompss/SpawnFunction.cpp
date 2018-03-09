/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6.h>
#include <nanos6/library-mode.h>

#include "SpawnFunction.hpp"
#include "lowlevel/SpinLock.hpp"

#include <cassert>
#include <map>
#include <mutex>
#include <string>
#include <utility>


class Task;


typedef std::pair<void (*)(void *), std::string> task_info_key_t;

static SpinLock _spawnedFunctionInfosLock;
static std::map<task_info_key_t, nanos_task_info> _spawnedFunctionInfos;

static nanos_task_invocation_info _spawnedFunctionInvocationInfo = { "Spawned from external code" };


namespace SpawnedFunctions {
	std::atomic<unsigned int> _pendingSpawnedFunctions(0);
}


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


static void nanos_spawned_function_wrapper(void *args, __attribute__((unused)) void *device_env, __attribute__((unused)) nanos6_address_translation_entry_t *translations)
{
	SpawnedFunctionArgsBlock *argsBlock = (SpawnedFunctionArgsBlock *) args;
	assert(argsBlock != nullptr);
	
	argsBlock->_function(argsBlock->_args);
	
	if (argsBlock->_completion_callback != nullptr) {
		argsBlock->_completion_callback(argsBlock->_completion_args);
	}
}


void nanos_spawn_function(void (*function)(void *), void *args, void (*completion_callback)(void *), void *completion_args, char const *label)
{
	SpawnedFunctions::_pendingSpawnedFunctions++;
	
	nanos_task_info *taskInfo = nullptr;
	{
		task_info_key_t taskInfoKey(function, (label != nullptr ? label : ""));
		
		std::lock_guard<SpinLock> guard(_spawnedFunctionInfosLock);
		auto itAndBool = _spawnedFunctionInfos.emplace( std::make_pair(taskInfoKey, nanos_task_info()) );
		auto it = itAndBool.first;
		taskInfo = &(it->second);
		taskInfo->implementations = malloc(sizeof(nanos6_task_implementation_info_t) * 1);
		
		if (itAndBool.second) {
			// New task info
			taskInfo->implementation_count = 1;
			taskInfo->implementations[0].run = nanos_spawned_function_wrapper;
			taskInfo->register_depinfo = nullptr;
			
			// We use the stored copy since we do not know the actual lifetime of "label"
			taskInfo->implementations[0].task_label = it->first.second.c_str();
			taskInfo->implementations[0].declaration_source = "Spawned Task";
			taskInfo->implementations[0].get_constraints = nullptr;
		}
	}
	
	SpawnedFunctionArgsBlock *argsBlock = nullptr;
	Task *task = nullptr;
	
	nanos_create_task(taskInfo, &_spawnedFunctionInvocationInfo, sizeof(SpawnedFunctionArgsBlock), (void **) &argsBlock, (void **) &task, 0);
	
	assert(argsBlock != nullptr);
	assert(task != nullptr);
	
	argsBlock->_function = function;
	argsBlock->_args = args;
	argsBlock->_completion_callback = completion_callback;
	argsBlock->_completion_args = completion_args;
	
	nanos_submit_task(task);
}

