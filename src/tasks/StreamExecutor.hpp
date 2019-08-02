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


class StreamExecutor : public Task {

private:
	
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
	
	inline StreamExecutor(
		void *argsBlock,
		size_t argsBlockSize,
		nanos6_task_info_t *taskInfo,
		nanos6_task_invocation_info_t *taskInvokationInfo,
		Task *parent,
		Instrument::task_id_t instrumentationTaskId,
		size_t flags
	)
		: Task(argsBlock, argsBlockSize, taskInfo, taskInvokationInfo, parent, instrumentationTaskId, flags),
		_blockingContext(nullptr),
		_mustShutdown(false),
		_queue(),
		_spinlock()
	{
	}
	
	inline ~StreamExecutor()
	{
		// Delete the executor's structures
		nanos6_task_info_t *executorInfo = getTaskInfo();
		assert(executorInfo != nullptr);
		assert(executorInfo->implementations != nullptr);
		free(executorInfo->implementations);
		free(executorInfo);
	}
	
	
	inline void setStreamId(size_t id)
	{
		_streamId = id;
	}
	
	inline size_t getStreamId() const
	{
		return _streamId;
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
				
				// Delete the executed function
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
