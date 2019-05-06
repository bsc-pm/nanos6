/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef SPAWN_FUNCTION_HPP
#define SPAWN_FUNCTION_HPP

#include <nanos6.h>

#include <atomic>


namespace SpawnedFunctions {
	extern std::atomic<unsigned int> _pendingSpawnedFunctions;
	
	//! \brief Indicates whether the task type is spawned
	//! 
	//! \param[in] taskInfo the task type
	//! 
	//! \returns true if it is a spawned task type
	bool isSpawned(const nanos6_task_info_t *taskInfo);
}


#endif // SPAWN_FUNCTION_HPP
