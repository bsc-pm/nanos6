#include "api/nanos6_library_interface.h"
#include "api/nanos6_rt_interface.h"
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


static void nanos_spawned_function_wrapper(void *args)
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
	nanos_task_info *taskInfo = nullptr;
	{
		task_info_key_t taskInfoKey(function, (label != nullptr ? label : ""));
		
		std::lock_guard<SpinLock> guard(_spawnedFunctionInfosLock);
		auto itAndBool = _spawnedFunctionInfos.emplace( std::make_pair(taskInfoKey, nanos_task_info()) );
		auto it = itAndBool.first;
		taskInfo = &(it->second);
		
		if (itAndBool.second) {
			// New task info
			taskInfo->run = nanos_spawned_function_wrapper;
			taskInfo->register_depinfo = nullptr;
			taskInfo->register_copies = nullptr;
			
			// We use the stored copy since we do not know the actual lifetime of "label"
			taskInfo->task_label = it->first.second.c_str();
			taskInfo->declaration_source = "Spawned Task";
			taskInfo->get_cost = nullptr;
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

