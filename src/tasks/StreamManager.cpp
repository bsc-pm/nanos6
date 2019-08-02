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
