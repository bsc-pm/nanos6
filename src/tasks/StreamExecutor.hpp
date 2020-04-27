/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
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
	void (*_callback)(void *);
	void *_callbackArgs;
	char const *_label;

	StreamFunction() :
		_function(nullptr),
		_args(nullptr),
		_callback(nullptr),
		_callbackArgs(nullptr),
		_label(nullptr)
	{
	}

	StreamFunction(
		void (*function)(void *),
		void *args,
		void (*callback)(void *),
		void *callbackArgs,
		char const *label
	) :
		_function(function),
		_args(args),
		_callback(callback),
		_callbackArgs(callbackArgs),
		_label(label)
	{
	}
};

struct StreamFunctionCallback {
	void (*_callback)(void *);
	void *_callbackArgs;
	std::atomic<size_t> _callbackParticipants;

	StreamFunctionCallback(
		void (*callback)(void *),
		void *callbackArgs,
		size_t callbackParticipants
	) :
		_callback(callback),
		_callbackArgs(callbackArgs),
		_callbackParticipants(callbackParticipants)
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

	//! Holds the callback of the function currently being executed
	StreamFunctionCallback *_currentCallback;

public:

	inline StreamExecutor(
		void *argsBlock,
		size_t argsBlockSize,
		nanos6_task_info_t *taskInfo,
		nanos6_task_invocation_info_t *taskInvokationInfo,
		Task *parent,
		Instrument::task_id_t instrumentationTaskId,
		size_t flags,
		TaskDataAccessesInfo taskAccessInfo,
		const TaskHardwareCounters &taskCounters
	)
		: Task(argsBlock, argsBlockSize, taskInfo, taskInvokationInfo, parent, instrumentationTaskId, flags, taskAccessInfo, taskCounters),
		_blockingContext(nullptr),
		_mustShutdown(false),
		_queue(),
		_spinlock(),
		_currentCallback(nullptr)
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

	//! \brief Increase the number of participants in a callback. This is so
	//! that the callback can be called when the last participant finishes
	//! \param[in] callback The pointer of the callback
	inline void increaseCallbackParticipants(StreamFunctionCallback *callback)
	{
		assert(callback != nullptr);

		callback->_callbackParticipants++;
	}

	//! \brief Decrease the number of participants in a callback. This is so
	//! that the callback can be called when the last participant finishes
	//! \param[in] callback The pointer of the callback
	inline void decreaseCallbackParticipants(StreamFunctionCallback *callback)
	{
		assert(callback != nullptr);

		if ((--(callback->_callbackParticipants)) == 0) {
			// If this is the last participant, execute and delete the callback
			callback->_callback(callback->_callbackArgs);
			delete callback;
		}
	}

	//! \brief Return the callback of the function being currently executed
	inline StreamFunctionCallback *getCurrentFunctionCallback() const
	{
		return _currentCallback;
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

				// If a callback exists for the function about to be executed,
				// register it in the map for a future trigger
				if (function->_callback != nullptr) {
					// The StreamExecutor in charge of the function participates
					// in the duty of calling the callback, hence by default
					// there's always one participant when creating callbacks
					StreamFunctionCallback *callbackObject = new StreamFunctionCallback(
						function->_callback,
						function->_callbackArgs,
						/* callbackParticipants = */ 1
					);

					_currentCallback = callbackObject;
				}

				// Execute the function
				function->_function(function->_args);

				// Decrease the participants of the callback of the executed
				// function, as this executor may need to execute the callback
				// if all child tasks have finished or none were created
				if (_currentCallback != nullptr) {
					decreaseCallbackParticipants(_currentCallback);
				}

				// Reset the pointer to the current callback
				_currentCallback = nullptr;

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
