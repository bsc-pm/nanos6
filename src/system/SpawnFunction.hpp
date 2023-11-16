/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef SPAWN_FUNCTION_HPP
#define SPAWN_FUNCTION_HPP

#include <nanos6.h>

#include <atomic>
#include <map>
#include <string>

#include "lowlevel/SpinLock.hpp"


class SpawnFunction {
public:
	//! Prototype of spawned functions and completion callbacks
	typedef void (*function_t)(void *args);

	//! Counter of pending spawned functions
	static std::atomic<unsigned int> _pendingSpawnedFunctions;

private:
	//! Key to identify spawned functions (user function and label)
	typedef std::pair<function_t, std::string> task_info_key_t;

	//! Map storing the task infos of spawned function
	static std::map<task_info_key_t, nanos6_task_info_t> _spawnedFunctionInfos;

	//! Spinlock to access the map task infos
	static SpinLock _spawnedFunctionInfosLock;

	//! Common invocation info for spawned functions
	static nanos6_task_invocation_info_t _spawnedFunctionInvocationInfo;

public:
	//! \brief Finalize the spawned functions
	static inline void shutdown()
	{
		for (auto spawned : _spawnedFunctionInfos) {
			nanos6_task_info_t &taskInfo = spawned.second;
			nanos6_task_implementation_info_t *implementations = taskInfo.implementations;
			assert(implementations != nullptr);
			free(implementations);
		}
	}

	//! \brief Spawn asynchronously a function
	//!
	//! \param[in] function The function to be spawned
	//! \param[in] args The parameter that is passed to the function
	//! \param[in] completionCallback An optional function that will be called when the function finishes
	//! \param[in] completionArgs The parameter that is passed to the completion callback
	//! \param[in] label An optional name for the function
	//! \param[in] fromUserCode Whether called from user code (i.e. nanos6_spawn_function)
	static void spawnFunction(
		function_t function,
		void *args,
		function_t completionCallback,
		void *completionArgs,
		char const *label,
		bool fromUserCode = false
	);

	//! \brief Indicates whether the task type is spawned
	//!
	//! \param[in] taskInfo The task type
	//!
	//! \returns Whether it is a spawned task type
	static bool isSpawned(const nanos6_task_info_t *taskInfo)
	{
		assert(taskInfo != nullptr);
		assert(taskInfo->implementations != nullptr);
		return (taskInfo->implementations[0].run == spawnedFunctionWrapper);
	}

private:
	//! \brief Wrapper function called by spawned tasks
	//!
	//! This function should call the function that was spawned
	//!
	//! \param[in,out] argsBlock The pointer to the block of data for the parameters
	//! \param[in] deviceEnv A pointer to device-specific data
	//! \param[in] translations One entry per task symbol that maps host addresses to device addresses
	static void spawnedFunctionWrapper(
		void *argsBlock,
		void *deviceEnv,
		nanos6_address_translation_entry_t *translations
	);

	//! \brief Function called when the spawned function has completed
	//!
	//! This function should call the completion callback of the spawned function
	//!
	//! \param[in,out] argsBlock A pointer to a block of data for the parameters
	static void spawnedFunctionDestructor(void *args);
};

#endif // SPAWN_FUNCTION_HPP
