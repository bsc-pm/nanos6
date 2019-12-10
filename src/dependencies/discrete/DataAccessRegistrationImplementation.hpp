/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_REGISTRATION_IMPLEMENTATION_HPP
#define DATA_ACCESS_REGISTRATION_IMPLEMENTATION_HPP


#include <mutex>

#include "DataAccessRegistration.hpp"
#include "TaskDataAccesses.hpp"
#include "tasks/Task.hpp"


namespace DataAccessRegistration {

	template <typename ProcessorType>
	inline bool processAllDataAccesses(Task *task, ProcessorType processor)
	{
		assert(task != nullptr);
		TaskDataAccesses &accessStruct = task->getDataAccesses();
		assert(!accessStruct.hasBeenDeleted());

		return accessStruct.forAll([&](void *, DataAccess *access) {
			return processor(access->getAccessRegion(), access->getType(), access->isWeak(), nullptr);
		});
	}

	template <typename ProcessorType>
	inline bool iterateAllDataAccesses(Task *task, ProcessorType processor)
	{
		assert(task != nullptr);
		TaskDataAccesses &accessStruct = task->getDataAccesses();
		assert(!accessStruct.hasBeenDeleted());

		return accessStruct.forAll([&](void *, DataAccess *access) {
			return processor(access);
		});
	}
}


#endif // DATA_ACCESS_REGISTRATION_IMPLEMENTATION_HPP
