#ifndef TASK_DATA_ACCESS_HOOKS_HPP
#define TASK_DATA_ACCESS_HOOKS_HPP

#include "TaskDataAccessLinkingArtifacts.hpp"


struct TaskDataAccessHooks {
	typedef typename TaskDataAccessLinkingArtifacts::hook_type TaskDataAccessesHook;
	
	TaskDataAccessesHook _accessesHook;
};


#endif // TASK_DATA_ACCESS_HOOKS_HPP
