/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "StreamManager.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


StreamManager *StreamManager::_manager;
nanos6_task_invocation_info_t StreamManager::_invocationInfo({"Spawned as a StreamExecutor"});
std::atomic<size_t> StreamManager::_activeStreamExecutors(0);


void StreamManager::createFunction(
	void (*function)(void *),
	void *args,
	void (*callback)(void *),
	void *callbackArgs,
	char const *label,
	size_t streamId
) {
	assert(_manager != nullptr);
	
	// Create the new stream function
	StreamFunction *streamFunction = new StreamFunction(function, args, callback, callbackArgs, label);
	assert(streamFunction != nullptr);
	
	// Add the new function to be executed in the stream
	StreamExecutor *executor = _manager->findOrCreateExecutor(streamId);
	executor->addFunction(streamFunction);
}

void StreamManager::synchronizeStream(size_t streamId)
{
	assert(_manager != nullptr);
	
	// Create a taskwait for the stream
	ConditionVariable condVar;
	StreamFunction *taskwait = new StreamFunction();
	assert(taskwait != nullptr);
	taskwait->_function = &(StreamExecutor::taskwaitBody);
	taskwait->_args = (void *) &condVar;
	
	// Add the taskwait as a new function to be executed in the stream
	StreamExecutor *executor = _manager->findOrCreateExecutor(streamId);
	executor->addFunction(taskwait);
	
	// Wait for the signal until the taskwait is completed
	condVar.wait();
}

void StreamManager::synchronizeAllStreams()
{
	assert(_manager != nullptr);
	
	size_t numTaskwaits = _manager->_executors.size();
	// Create an array of taskwaits and an array of condition variables
	StreamFunction *taskwaits[numTaskwaits];
	ConditionVariable condVars[numTaskwaits];
	size_t index = 0;
	
	// Initialize a taskwait and a condition variable for every executor
	for (auto &it : _manager->_executors) {
		taskwaits[index] = new StreamFunction();
		StreamFunction *taskwait = taskwaits[index];
		assert(taskwait != nullptr);
		taskwait->_function = &(StreamExecutor::taskwaitBody);
		taskwait->_args = (void *) &(condVars[index]);
		
		StreamExecutor *executor = it.second;
		assert(executor != nullptr);
		executor->addFunction(taskwait);
		++index;
	}
	
	// Wait for all taskwaits to end
	for (size_t i = 0; i < numTaskwaits; ++i) {
		// Wait for the signal until the taskwait is completed
		condVars[i].wait();
	}
	
	// NOTE: The dynamically created StreamFunction-taskwaits are deleted upon
	// completion by the appropriate StreamExecutor (see StreamExecutor::body)
}
