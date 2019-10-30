/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_REGISTRATION_IMPLEMENTATION_HPP
#define DATA_ACCESS_REGISTRATION_IMPLEMENTATION_HPP


#include <mutex>

#include "DataAccessRegistration.hpp"
#include "tasks/Task.hpp"


namespace DataAccessRegistration {

	// Placeholder
	template <typename ProcessorType>
	inline bool processAllDataAccesses(__attribute__((unused)) Task *task, __attribute__((unused)) ProcessorType processor)
	{
		return true;
	}
}


#endif // DATA_ACCESS_REGISTRATION_IMPLEMENTATION_HPP
