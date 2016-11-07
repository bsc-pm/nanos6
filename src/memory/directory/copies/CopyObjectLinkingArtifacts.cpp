#include <boost/intrusive/parent_from_member.hpp>

#include "CopyObject.hpp"
#include "TaskDataAccessLinkingArtifacts.hpp"


inline constexpr CopyObjectLinkingArtifacts::hook_ptr
CopyObjectLinkingArtifacts::to_hook_ptr (CopyObjectLinkingArtifacts::value_type &value)
{
	return &value._hook;
}

inline constexpr CopyObjectLinkingArtifacts::const_hook_ptr
CopyObjectLinkingArtifacts::to_hook_ptr(const CopyObjectLinkingArtifacts::value_type &value)
{
	return &value._hook;
}

inline CopyObjectLinkingArtifacts::pointer
CopyObjectLinkingArtifacts::to_value_ptr(CopyObjectLinkingArtifacts::hook_ptr n)
{
	return (CopyObjectLinkingArtifacts::pointer)
		boost::intrusive::get_parent_from_member<CopyObject>(
			n,
			&CopyObject::_hook
		);
}

inline TaskDataAccessLinkingArtifacts::const_pointer
TaskDataAccessLinkingArtifacts::to_value_ptr(TaskDataAccessLinkingArtifacts::const_hook_ptr n)
{
	return (TaskDataAccessLinkingArtifacts::const_pointer)
		boost::intrusive::get_parent_from_member<CopyObject>(
			n,
			&CopyObject::_hook
			)
		;
}

