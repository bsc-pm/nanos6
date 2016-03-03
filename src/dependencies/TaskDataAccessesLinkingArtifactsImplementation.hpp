#ifndef TASK_DATA_ACCESSES_LINKING_ARTIFACTS_IMPLEMENTATION_HPP
#define TASK_DATA_ACCESSES_LINKING_ARTIFACTS_IMPLEMENTATION_HPP

#include <boost/intrusive/parent_from_member.hpp>

#include "DataAccessBase.hpp"
#include "TaskDataAccessesLinkingArtifacts.hpp"


inline constexpr TaskDataAccessesLinkingArtifacts::hook_ptr TaskDataAccessesLinkingArtifacts::to_hook_ptr (TaskDataAccessesLinkingArtifacts::value_type &value)
{
	return &value._taskAccessListLinks;
}

inline constexpr TaskDataAccessesLinkingArtifacts::const_hook_ptr TaskDataAccessesLinkingArtifacts::to_hook_ptr(const TaskDataAccessesLinkingArtifacts::value_type &value)
{
	return &value._taskAccessListLinks;
}

inline TaskDataAccessesLinkingArtifacts::pointer TaskDataAccessesLinkingArtifacts::to_value_ptr(TaskDataAccessesLinkingArtifacts::hook_ptr n)
{
	return boost::intrusive::get_parent_from_member<DataAccessBase>(n, &DataAccessBase::_taskAccessListLinks);
}

inline TaskDataAccessesLinkingArtifacts::const_pointer TaskDataAccessesLinkingArtifacts::to_value_ptr(TaskDataAccessesLinkingArtifacts::const_hook_ptr n)
{
	return boost::intrusive::get_parent_from_member<DataAccessBase>(n, &DataAccessBase::_taskAccessListLinks);
}


#endif // TASK_DATA_ACCESSES_LINKING_ARTIFACTS_IMPLEMENTATION_HPP
