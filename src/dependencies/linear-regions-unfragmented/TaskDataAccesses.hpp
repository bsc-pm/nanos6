#ifndef TASK_DATA_ACCESSES_HPP
#define TASK_DATA_ACCESSES_HPP


#include <boost/intrusive/list.hpp>

#include "TaskDataAccessLinkingArtifacts.hpp"


struct DataAccess;


typedef boost::intrusive::list<
	DataAccess,
	boost::intrusive::function_hook<TaskDataAccessLinkingArtifacts>
> TaskDataAccesses;

typedef typename TaskDataAccessLinkingArtifacts::hook_type TaskDataAccessHooks;


#endif // TASK_DATA_ACCESSES_HPP
