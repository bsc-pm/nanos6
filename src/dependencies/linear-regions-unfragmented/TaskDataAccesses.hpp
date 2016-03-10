#ifndef TASK_DATA_ACCESSES_HPP
#define TASK_DATA_ACCESSES_HPP


#include <boost/intrusive/list.hpp>

#include "TaskDataAccessesLinkingArtifacts.hpp"


struct DataAccess;


typedef boost::intrusive::list<
	DataAccess,
	boost::intrusive::function_hook<TaskDataAccessesLinkingArtifacts>
> TaskDataAccesses;

typedef typename TaskDataAccessesLinkingArtifacts::hook_type TaskDataAccessesHook;


#endif // TASK_DATA_ACCESSES_HPP
