/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_REGISTRATION_IMPLEMENTATION_HPP
#define DATA_ACCESS_REGISTRATION_IMPLEMENTATION_HPP


#include <mutex>

#include "DataAccessRegistration.hpp"
#include "TaskDataAccesses.hpp"
#include "tasks/Task.hpp"


namespace DataAccessRegistration {

	// Placeholder
	template <typename ProcessorType>
	inline bool processAllDataAccesses(Task *task, ProcessorType processor)
	{
		assert(task != nullptr);
		TaskDataAccesses &accessStruct = task->getDataAccesses();
		assert(!accessStruct.hasBeenDeleted());

		DataAccess * accessArray = accessStruct._accessArray;
		void ** addressArray = accessStruct._addressArray;

		for (size_t i = 0; i < accessStruct.getRealAccessNumber(); ++i) {
			DataAccess * access = &accessArray[i];
			void * address = addressArray[i];

			DataAccessRegion region(address, access->getLength());
			if(!processor(region, access->getType(), access->isWeak(), nullptr))
				return false;
		}

		return true;
	}
}


#endif // DATA_ACCESS_REGISTRATION_IMPLEMENTATION_HPP
