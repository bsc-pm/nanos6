/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef STREAM_MANAGER_HPP
#define STREAM_MANAGER_HPP

#include <map>

#include <nanos6.h>

#include "StreamExecutor.hpp"
#include "system/ompss/AddTask.hpp"
#include "system/ompss/SpawnFunction.hpp"
#include "tasks/TaskImplementation.hpp"


class StreamManager {

private:

	typedef std::map<size_t, StreamExecutor *> stream_executors_t;

	//! Singleton instance
	static StreamManager *_manager;

	//! Maps stream executors through their stream identifier
	stream_executors_t _executors;

	//! Spinlock to add new stream executors and access existent ones
	SpinLock _spinlock;

	//! A static invocation info object for all Stream Executors
	static nanos6_task_invocation_info_t _invocationInfo;


public:

	//! The amount of active stream executors
	static std::atomic<size_t> _activeStreamExecutors;


private:

	inline StreamManager() :
		_executors(),
		_spinlock()
	{
	}

	//! \brief Find or create a stream executor
	//! \param[in] streamId The id of the stream the executor is in charge of
	//! \return A pointer to the stream executor in charge of streamId
	StreamExecutor *findOrCreateExecutor(size_t streamId)
	{
		StreamExecutor *executor;

		_spinlock.lock();

		stream_executors_t::iterator it = _executors.find(streamId);
		if (it == _executors.end()) {
			// Executor's taskinfo
			// Executor's args block
			nanos6_task_info_t *executorInfo = (nanos6_task_info_t *) malloc(sizeof(nanos6_task_info_t));
			assert(executorInfo != nullptr);

			// Fill in the executor's taskinfo
			executorInfo->implementations = (nanos6_task_implementation_info_t *)
				malloc(sizeof(nanos6_task_implementation_info_t) * 1);
			assert(executorInfo->implementations != nullptr);

			executorInfo->implementation_count = 1;
			executorInfo->implementations[0].run = &(StreamExecutor::bodyWrapper);
			executorInfo->implementations[0].device_type_id = nanos6_device_t::nanos6_host_device;
			executorInfo->implementations[0].task_type_label = "StreamExecutor";
			executorInfo->implementations[0].declaration_source = "Stream Executor spawned within the runtime";
			executorInfo->implementations[0].get_constraints = nullptr;
			executorInfo->num_symbols = 0;
			executorInfo->destroy_args_block = nullptr;
			executorInfo->register_depinfo = nullptr;
			executorInfo->get_priority = nullptr;
			executorInfo->onready_action = nullptr;
			executorInfo->duplicate_args_block = nullptr;
			executorInfo->reduction_initializers = nullptr;
			executorInfo->reduction_combiners = nullptr;
			executorInfo->task_type_data = nullptr;
			executorInfo->iter_condition = nullptr;
			executorInfo->num_args = 0;
			executorInfo->sizeof_table = nullptr;
			executorInfo->offset_table = nullptr;
			executorInfo->arg_idx_table = nullptr;
			size_t flags = 1 << Task::stream_executor_flag;

			// Create the Stream Executor task
			executor = (StreamExecutor *) AddTask::createTask(
				executorInfo, &_invocationInfo,
				nullptr, sizeof(StreamExecutorArgsBlock),
				flags
			);
			assert(executor != nullptr);

			StreamExecutorArgsBlock *argsBlock =
				(StreamExecutorArgsBlock *) executor->getArgsBlock();
			assert(argsBlock != nullptr);

			// Complete the args block of the executor
			// Pass itself as arguments to access the body
			argsBlock->_executor = (void *) executor;

			// Set the identifier of the stream the executor is in charge of
			executor->setStreamId(streamId);

			// Emplace the executor in the executors map
			_executors.emplace(std::make_pair(streamId, executor));

			// Release the lock as it is no longer needed
			_spinlock.unlock();

			// Increase the number of active stream executors
			++_activeStreamExecutors;

			// Submit the executor without parent
			AddTask::submitTask(executor, nullptr);
		} else {
			executor = it->second;

			// Release the lock as it is no longer needed
			_spinlock.unlock();
		}

		return executor;
	}


public:

	// Delete copy and move constructors/assign operators
	StreamManager(StreamManager const&) = delete;            // Copy construct
	StreamManager(StreamManager&&) = delete;                 // Move construct
	StreamManager& operator=(StreamManager const&) = delete; // Copy assign
	StreamManager& operator=(StreamManager &&) = delete;     // Move assign


	//    MANAGER    //

	//! \brief Initialize the manager
	static inline void initialize()
	{
		if (_manager == nullptr) {
			_manager = new StreamManager();
			assert(_manager != nullptr);
		}
	}

	//! \brief Shutdown mechanism for all the stream executors
	static inline void shutdown()
	{
		if (_manager != nullptr) {
			_manager->_spinlock.lock();

			// Notify all executors about the shutdown
			for (auto &it : _manager->_executors) {
				StreamExecutor *executor = it.second;
				assert(executor != nullptr);
				executor->notifyShutdown();
			}

			_manager->_spinlock.unlock();

			while (_activeStreamExecutors.load() > 0) {
				// Wait for all active stream executors to finalize
			}

			delete _manager;
		}
	}


	//    STREAM HANDLING    //

	//! \brief Create a function to be executed in a stream
	//! \param[in] function The function to execute
	//! \param[in] args Arguments of the function
	//! \param[in] callback An optional callback called upon completion
	//! \param[in] callbackArgs Parameters passed to the callback
	//! \param[in] label An optional label for the function
	//! \param[in] streamId The identifier of the stream
	static void createFunction(
		void (*function)(void *),
		void *args,
		void (*callback)(void *),
		void *callbackArgs,
		char const *label,
		size_t streamId
	);

	//! \brief Synchronize (block control flow) of a certain stream until
	//! all spawned functions have finalized
	//! \param[in] streamId The identifier of the stream to synchronize
	static void synchronizeStream(size_t streamId);

	//! \brief Create taskwaits for all streams
	static void synchronizeAllStreams();

};

#endif // STREAM_MANAGER_HPP
