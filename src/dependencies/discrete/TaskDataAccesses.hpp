/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

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
