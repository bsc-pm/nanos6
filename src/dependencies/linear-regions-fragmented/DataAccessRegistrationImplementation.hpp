/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
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
				return processor(&access);
			}
		);
	}

	inline void updateTaskDataAccessLocation(Task *task,
			DataAccessRegion const &region, MemoryPlace const *location,
			bool isTaskwait)
	{
		assert(task != nullptr);

		TaskDataAccesses &accessStructures = task->getDataAccesses();
		assert(!accessStructures.hasBeenDeleted());

		std::lock_guard<TaskDataAccesses::spinlock_t> guard(accessStructures._lock);

		auto &accesses = (isTaskwait) ?
			accessStructures._taskwaitFragments :
			accessStructures._accesses;

		// At this point the region must be included in DataAccesses of the task
		assert(accesses.contains(region));

		accesses.processIntersecting(region,
			[&](TaskDataAccesses::accesses_t::iterator accessPosition) -> bool {
				DataAccess *access = &(*accessPosition);
				assert(access != nullptr);
				assert(access->getAccessRegion().fullyContainedIn(region));

				access->setLocation(location);

				return true;
			}
		);
	}
}


#endif // DATA_ACCESS_REGISTRATION_IMPLEMENTATION_HPP
