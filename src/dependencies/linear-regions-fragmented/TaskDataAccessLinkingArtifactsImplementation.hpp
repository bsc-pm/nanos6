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


inline constexpr TaskDataSubaccessBottomMapLinkingArtifacts::hook_ptr
TaskDataSubaccessBottomMapLinkingArtifacts::to_hook_ptr (TaskDataSubaccessBottomMapLinkingArtifacts::value_type &value)
{
	return &value._taskDataAccessLinks._subaccessBottomMapHook;
}

inline constexpr TaskDataSubaccessBottomMapLinkingArtifacts::const_hook_ptr
TaskDataSubaccessBottomMapLinkingArtifacts::to_hook_ptr(const TaskDataSubaccessBottomMapLinkingArtifacts::value_type &value)
{
	return &value._taskDataAccessLinks._subaccessBottomMapHook;
}

inline TaskDataSubaccessBottomMapLinkingArtifacts::pointer
TaskDataSubaccessBottomMapLinkingArtifacts::to_value_ptr(TaskDataSubaccessBottomMapLinkingArtifacts::hook_ptr n)
{
	return (TaskDataSubaccessBottomMapLinkingArtifacts::pointer)
		boost::intrusive::get_parent_from_member<DataAccessBase>(
			boost::intrusive::get_parent_from_member<TaskDataAccessHooks>(
				n,
				&TaskDataAccessHooks::_subaccessBottomMapHook
			),
			&DataAccessBase::_taskDataAccessLinks
		);
}

inline TaskDataSubaccessBottomMapLinkingArtifacts::const_pointer
TaskDataSubaccessBottomMapLinkingArtifacts::to_value_ptr(TaskDataSubaccessBottomMapLinkingArtifacts::const_hook_ptr n)
{
	return (TaskDataSubaccessBottomMapLinkingArtifacts::const_pointer)
		boost::intrusive::get_parent_from_member<DataAccessBase>(
			boost::intrusive::get_parent_from_member<TaskDataAccessHooks>(
				n,
				&TaskDataAccessHooks::_subaccessBottomMapHook
			),
			&DataAccessBase::_taskDataAccessLinks
		);
}


#endif // TASK_DATA_ACCESS_LINKING_ARTIFACTS_IMPLEMENTATION_HPP
