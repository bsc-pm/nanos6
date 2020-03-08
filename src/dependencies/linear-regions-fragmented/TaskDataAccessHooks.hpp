/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_DATA_ACCESS_HOOKS_HPP
#define TASK_DATA_ACCESS_HOOKS_HPP

#include "TaskDataAccessLinkingArtifacts.hpp"


struct TaskDataAccessHooks {
	typedef typename TaskDataAccessLinkingArtifacts::hook_type TaskDataAccessesHook;
	
	TaskDataAccessesHook _accessesHook;
};


#endif // TASK_DATA_ACCESS_HOOKS_HPP
