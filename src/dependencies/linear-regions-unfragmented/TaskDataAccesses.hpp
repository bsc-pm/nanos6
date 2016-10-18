#ifndef TASK_DATA_ACCESSES_HPP
#define TASK_DATA_ACCESSES_HPP


#include <boost/intrusive/list.hpp>

#include "TaskDataAccessLinkingArtifacts.hpp"


struct DataAccess;


struct TaskDataAccesses
	: public boost::intrusive::list<
		DataAccess,
		boost::intrusive::function_hook<TaskDataAccessLinkingArtifacts>
	>
{
};


typedef typename TaskDataAccessLinkingArtifacts::hook_type TaskDataAccessHooks;


#endif // TASK_DATA_ACCESSES_HPP
