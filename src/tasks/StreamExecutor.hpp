/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef STREAM_EXECUTOR_HPP
#define STREAM_EXECUTOR_HPP

#include <deque>
#include <pthread.h>

#include <nanos6.h>

#include "lowlevel/ConditionVariable.hpp"
#include "tasks/Task.hpp"
#include "system/ompss/SpawnFunction.hpp"


struct StreamFunction {
	void (*_function)(void *);
	void *_args;
	char const *_label;
	
	StreamFunction() :
		_function(nullptr),
		_args(nullptr),
		_label(nullptr)
	{
	}
	
	StreamFunction(void (*function)(void *), void *args, char const *label) :
		_function(function),
		_args(args),
		_label(label)
	{
	}
};

struct StreamExecutorArgsBlock {
	void *_executor;
	
	StreamExecutorArgsBlock() :
		_executor(nullptr)
	{
	}
};


class StreamExecutor {

private:
	
	//! The executor itself, a task that executes functions from a stream
	Task *_executor;
	
	//! The identifier of the stream this executor is in charge of
	size_t _streamId;
	
	//! The blocking context of the executor task
	void *_blockingContext;
	
	//! Whether the runtime is shutting down
	std::atomic<bool> _mustShutdown;
	
	//! The executor's function queue
	std::deque<StreamFunction *> _queue;
	
	//! A spinlock to access the queue and block/unblock the executor
	SpinLock _spinlock;
	
	
public:
	
	inline StreamExecutor(size_t id) :
		_executor(nullptr),
		_streamId(id),
		_blockingContext(nullptr),
		_mustShutdown(false),
		_queue(),
		_spinlock()
	{
		// The executor's taskinfo's invocation information
		// The executor's taskinfo
		// The executor's args block
		nanos6_task_invocation_info_t *executorInvocationInfo = new nanos6_task_invocation_info_t();
		nanos6_task_info_t *executorInfo = new nanos6_task_info_t();
		assert(executorInfo != nullptr);
		assert(executorInvocationInfo != nullptr);
		StreamExecutorArgsBlock *argsBlock;
		
		executorInfo->implementations = (nanos6_task_implementation_info_t *) malloc(sizeof(nanos6_task_implementation_info_t));
		assert(executorInfo->implementations != nullptr);
		executorInfo->implementation_count = 1;
		executorInfo->implementations[0].run = &(StreamExecutor::bodyWrapper);
		executorInfo->implementations[0].device_type_id = nanos6_device_t::nanos6_host_device;
		executorInfo->implementations[0].task_label = "StreamExecutor";
		executorInfo->implementations[0].declaration_source = "";
		executorInfo->implementations[0].get_constraints = nullptr;
		executorInfo->num_symbols = 0;
		executorInfo->register_depinfo = nullptr;
		executorInfo->destroy_args_block = nullptr;
		executorInfo->get_priority = nullptr;
		
		// Create the executor task
		nanos6_create_task(
			executorInfo,
			executorInvocationInfo,
			sizeof(StreamExecutorArgsBlock),
			(void **) &argsBlock,
			(void **) &_executor,
			1 << Task::stream_executor_flag,
			0
		);
		
		assert(argsBlock != nullptr);
		assert(_executor != nullptr);
		
		// Complete the args block of the executor
		// Pass itself as arguments to access the body
		argsBlock->_executor = (void *) _executor;
		
		// Submit the executor to the scheduler
		nanos6_submit_task(_executor);
	}
	
	
	//! \brief Notify to the executor that it must be shutdown
	inline void notifyShutdown()
	{
		_spinlock.lock();
		
		_mustShutdown = true;
		void *blockingContext = _blockingContext;
		_blockingContext = nullptr;
		
		_spinlock.unlock();
		
		// Unblock the executor if it was blocked
		if (blockingContext != nullptr) {
			nanos6_unblock_task(blockingContext);
		}
		
		// Delete the executor's structures
		nanos6_task_invocation_info_t *executorInvocationInfo = _executor->getTaskInvokationInfo();
		nanos6_task_info_t *executorInfo = _executor->getTaskInfo();
		assert(executorInvocationInfo != nullptr);
		assert(executorInfo != nullptr);
		delete executorInvocationInfo;
		delete executorInfo;
	}
	
	//! \brief Add a function to this executor's stream queue
	//! \param[in] function The kernel to execute
	inline void addFunction(StreamFunction *function)
	{
		_spinlock.lock();
		
		_queue.push_back(function);
		void *blockingContext = _blockingContext;
		_blockingContext = nullptr;
		
		_spinlock.unlock();
		
		// Unblock the executor if it was blocked
		if (blockingContext != nullptr) {
			nanos6_unblock_task(blockingContext);
		}
	}
	
	//! \brief The body of a stream executor
	inline void body()
	{
		StreamFunction *function;
		while (!_mustShutdown.load()) {
			function = nullptr;
			_spinlock.lock();
			
			if (!_queue.empty()) {
				// Get the first function in the stream's queue
				function = _queue.front();
				assert(function != nullptr);
				_queue.pop_front();
				
				_spinlock.unlock();
				
				// Execute the function
				function->_function(function->_args);
				delete function;
			} else {
				// Release the lock and block the task
				assert(_blockingContext == nullptr);
				_blockingContext = nanos6_get_current_blocking_context();
				void *blockingContext = _blockingContext;
				
				_spinlock.unlock();
				
				nanos6_block_current_task(blockingContext);
			}
		}
	}
	
	
	//    STATIC WRAPPERS    //
	
	//! \brief A wrapper that executes the body of a Stream Executor
	//! \param args The arguments of the executor, which is a pointer
	//! to the StreamExecutor to access its methods (the main body)
	static void bodyWrapper(void *args, void *, nanos6_address_translation_entry_t *)
	{
		StreamExecutorArgsBlock *argsBlock = (StreamExecutorArgsBlock *) args;
		assert(argsBlock != nullptr);
		StreamExecutor *executor = (StreamExecutor *) argsBlock->_executor;
		assert(executor != nullptr);
		
		executor->body();
	}
	
	//! \brief The body of a taskwait for a Stream Executor
	//! \param args A pointer to the condition variable of the taskwait
	static void taskwaitBody(void *args)
	{
		nanos6_taskwait("");
		
		ConditionVariable *condVar = (ConditionVariable *) args;
		assert(condVar != nullptr);
		
		// Signal that the taskwait has been completed
		condVar->signal();
	}
	
};

#endif // STREAM_EXECUTOR_HPP
