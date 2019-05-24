/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_REGISTRATION_IMPLEMENTATION_HPP
#define DATA_ACCESS_REGISTRATION_IMPLEMENTATION_HPP


#include <mutex>

#include "DataAccess.hpp"
#include "DataAccessRegistration.hpp"
#include "TaskDataAccesses.hpp"
#include "tasks/Task.hpp"


namespace DataAccessRegistration {
	
	template <typename ProcessorType>
	inline bool processAllDataAccesses(Task *task, ProcessorType processor)
	{
		assert(task != nullptr);
		
		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());
		
		std::lock_guard<TaskDataAccesses::spinlock_t> guard(accessStructures._lock);
		
		return accessStructures._accesses.processAll(
			[&](TaskDataAccesses::accesses_t::iterator position) -> bool {
				DataAccess &access = *position;
				return processor(access.getAccessRegion(), access.getType(), access.isWeak(), access.getLocation());
			}
		);
	}
}


#endif // DATA_ACCESS_REGISTRATION_IMPLEMENTATION_HPP
