/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "StreamManager.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


StreamManager *StreamManager::_manager;
std::atomic<size_t> StreamManager::_activeStreamExecutors(0);


void StreamManager::createFunction(
	void (*function)(void *),
	void *args,
	char const *label,
	size_t streamId
) {
	assert(_manager != nullptr);
	
	// Create the new stream function
	StreamFunction *streamFunction = new StreamFunction(function, args, label);
	assert(streamFunction != nullptr);
	
	StreamExecutor *executor;
	
	_manager->_spinlock.lock();
	
	// Find or create a stream executor with 'streamId' as identifier
	stream_executors_t::iterator it = _manager->_executors.find(streamId);
	if (it == _manager->_executors.end()) {
		executor = new StreamExecutor(streamId);
		_manager->_executors.emplace(std::make_pair(streamId, executor));
		
		// Increase the number of active stream executors
		++_activeStreamExecutors;
	} else {
		executor = it->second;
	}
	
	_manager->_spinlock.unlock();
	
	// Add the new function to be executed in the stream
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
	
	StreamExecutor *executor;
	
	_manager->_spinlock.lock();
	
	// Find or create a stream executor with 'streamId' as identifier
	stream_executors_t::iterator it = _manager->_executors.find(streamId);
	if (it == _manager->_executors.end()) {
		executor = new StreamExecutor(streamId);
		_manager->_executors.emplace(std::make_pair(streamId, executor));
		
		// Increase the number of active stream executors
		++_activeStreamExecutors;
	} else {
		executor = it->second;
	}
	
	_manager->_spinlock.unlock();
	
	// Add the taskwait as a new function to be executed in the stream
	executor->addFunction(taskwait);
	
	// Wait for the signal until the taskwait is completed
	condVar.wait();
}
