/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef SPAWN_FUNCTION_HPP
#define SPAWN_FUNCTION_HPP

#include <atomic>


namespace SpawnedFunctions {
	extern std::atomic<unsigned int> _pendingSpawnedFunctions;
}


#endif // SPAWN_FUNCTION_HPP
