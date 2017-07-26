/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_DATA_ACCESS_LINKING_ARTIFACTS_IMPLEMENTATION_HPP
#define TASK_DATA_ACCESS_LINKING_ARTIFACTS_IMPLEMENTATION_HPP

#include <boost/intrusive/parent_from_member.hpp>

#include "DataAccess.hpp"
#include "TaskDataAccessLinkingArtifacts.hpp"


inline constexpr TaskDataAccessLinkingArtifacts::hook_ptr
TaskDataAccessLinkingArtifacts::to_hook_ptr (TaskDataAccessLinkingArtifacts::value_type &value)
{
	return &value._taskDataAccessLinks._accessesHook;
}

inline constexpr TaskDataAccessLinkingArtifacts::const_hook_ptr
TaskDataAccessLinkingArtifacts::to_hook_ptr(const TaskDataAccessLinkingArtifacts::value_type &value)
{
	return &value._taskDataAccessLinks._accessesHook;
}

inline TaskDataAccessLinkingArtifacts::pointer
TaskDataAccessLinkingArtifacts::to_value_ptr(TaskDataAccessLinkingArtifacts::hook_ptr n)
{
	return (TaskDataAccessLinkingArtifacts::pointer)
		boost::intrusive::get_parent_from_member<DataAccessBase>(
			boost::intrusive::get_parent_from_member<TaskDataAccessHooks>(
				n,
				&TaskDataAccessHooks::_accessesHook
			),
			&DataAccessBase::_taskDataAccessLinks
		);
}

inline TaskDataAccessLinkingArtifacts::const_pointer
TaskDataAccessLinkingArtifacts::to_value_ptr(TaskDataAccessLinkingArtifacts::const_hook_ptr n)
{
	return (TaskDataAccessLinkingArtifacts::const_pointer)
		boost::intrusive::get_parent_from_member<DataAccessBase>(
			boost::intrusive::get_parent_from_member<TaskDataAccessHooks>(
				n,
				&TaskDataAccessHooks::_accessesHook
			),
			&DataAccessBase::_taskDataAccessLinks
		);
}


#endif // TASK_DATA_ACCESS_LINKING_ARTIFACTS_IMPLEMENTATION_HPP
