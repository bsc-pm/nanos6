#ifndef TASK_DATA_ACCESSES_LINKING_ARTIFACTS_IMPLEMENTATION_HPP
#define TASK_DATA_ACCESSES_LINKING_ARTIFACTS_IMPLEMENTATION_HPP

#include <boost/intrusive/parent_from_member.hpp>

#include "DataAccessSequence.hpp"
#include "DataAccess.hpp"

#include "TaskDataAccessesLinkingArtifacts.hpp"
#include "DataAccessSequenceLinkingArtifacts.hpp"


inline constexpr TaskDataAccessesLinkingArtifacts::hook_ptr TaskDataAccessesLinkingArtifacts::to_hook_ptr (TaskDataAccessesLinkingArtifacts::value_type &value)
{
	return &value._taskDataAccessesLinks;
}

inline constexpr TaskDataAccessesLinkingArtifacts::const_hook_ptr TaskDataAccessesLinkingArtifacts::to_hook_ptr(const TaskDataAccessesLinkingArtifacts::value_type &value)
{
	return &value._taskDataAccessesLinks;
}

inline TaskDataAccessesLinkingArtifacts::pointer TaskDataAccessesLinkingArtifacts::to_value_ptr(TaskDataAccessesLinkingArtifacts::hook_ptr n)
{
	return (TaskDataAccessesLinkingArtifacts::pointer) boost::intrusive::get_parent_from_member<DataAccessBase>(n, &DataAccessBase::_taskDataAccessesLinks);
}

inline TaskDataAccessesLinkingArtifacts::const_pointer TaskDataAccessesLinkingArtifacts::to_value_ptr(TaskDataAccessesLinkingArtifacts::const_hook_ptr n)
{
	return (TaskDataAccessesLinkingArtifacts::const_pointer) boost::intrusive::get_parent_from_member<DataAccessBase>(n, &DataAccessBase::_taskDataAccessesLinks);
}


#endif // TASK_DATA_ACCESSES_LINKING_ARTIFACTS_IMPLEMENTATION_HPP
