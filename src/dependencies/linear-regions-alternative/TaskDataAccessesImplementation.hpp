#ifndef TASK_DATA_ACCESSES_IMPLEMENTATION_HPP
#define TASK_DATA_ACCESSES_IMPLEMENTATION_HPP

#include <boost/intrusive/parent_from_member.hpp>

#include <InstrumentDependenciesByAccessLinks.hpp>

#include "BottomMapEntry.hpp"
#include "DataAccess.hpp"
#include "TaskDataAccesses.hpp"
#include "tasks/Task.hpp"


inline TaskDataAccesses::~TaskDataAccesses()
{
	assert(!hasBeenDeleted());
	
	Task *task = boost::intrusive::get_parent_from_member<Task>(this, &Task::_dataAccesses);
	assert(task != nullptr);
	assert(&task->getDataAccesses() == this);
	
	assert(_removalCountdown == 0);
	assert(_accesses.empty());
	assert(_subaccessBottomMap.empty());
	
	#ifndef NDEBUG
	hasBeenDeleted() = true;
	#endif
}


#endif // TASK_DATA_ACCESSES_IMPLEMENTATION_HPP
