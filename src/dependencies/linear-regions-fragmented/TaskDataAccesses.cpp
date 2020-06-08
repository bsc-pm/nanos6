/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include <boost/intrusive/parent_from_member.hpp>

#include "BottomMapEntry.hpp"
#include "DataAccess.hpp"
#include "ObjectAllocator.hpp"
#include "TaskDataAccesses.hpp"
#include "TaskDataAccessLinkingArtifacts.hpp"
#include "tasks/Task.hpp"

TaskDataAccesses::~TaskDataAccesses()
{
	assert(!hasBeenDeleted());
	assert(_removalBlockers == 0);
	
#ifndef NDEBUG
	Task *task = boost::intrusive::get_parent_from_member<Task>(this, &Task::_dataAccesses);
	assert(task != nullptr);
	assert(&task->getDataAccesses() == this);
#endif
	
	// We take the lock since the task may be marked for deletion while the lock is held
	std::lock_guard<spinlock_t> guard(_lock);
	_accesses.deleteAll(
		[&](DataAccess *access) {
			ObjectAllocator<DataAccess>::deleteObject(access);
		}
	);
	
	_subaccessBottomMap.deleteAll(
		[&](BottomMapEntry *bottomMapEntry) {
			ObjectAllocator<BottomMapEntry>::deleteObject(bottomMapEntry);
		}
	);
	
	_accessFragments.deleteAll(
		[&](DataAccess *fragment) {
			ObjectAllocator<DataAccess>::deleteObject(fragment);
		}
	);
	
	_taskwaitFragments.deleteAll(
		[&](DataAccess *fragment) {
			ObjectAllocator<DataAccess>::deleteObject(fragment);
		}
	);
	
#ifndef NDEBUG
	hasBeenDeleted() = true;
#endif
}
