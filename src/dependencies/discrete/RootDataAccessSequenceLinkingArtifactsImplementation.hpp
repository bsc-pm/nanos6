/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef ROOT_DATA_ACCESS_SEQUENCE_LINKING_ARTIFACTS_IMPLEMENTATION_HPP
#define ROOT_DATA_ACCESS_SEQUENCE_LINKING_ARTIFACTS_IMPLEMENTATION_HPP


#include <boost/intrusive/parent_from_member.hpp>

#include "RootDataAccessSequence.hpp"
#include "RootDataAccessSequenceLinkingArtifacts.hpp"


inline constexpr RootDataAccessSequenceLinkingArtifacts::hook_ptr
RootDataAccessSequenceLinkingArtifacts::to_hook_ptr (RootDataAccessSequenceLinkingArtifacts::value_type &value)
{
	return &value._rootDataAccessSequenceLinks;
}

inline constexpr RootDataAccessSequenceLinkingArtifacts::const_hook_ptr
RootDataAccessSequenceLinkingArtifacts::to_hook_ptr(const RootDataAccessSequenceLinkingArtifacts::value_type &value)
{
	return &value._rootDataAccessSequenceLinks;
}

inline RootDataAccessSequenceLinkingArtifacts::pointer
RootDataAccessSequenceLinkingArtifacts::to_value_ptr(RootDataAccessSequenceLinkingArtifacts::hook_ptr n)
{
	return (RootDataAccessSequenceLinkingArtifacts::pointer)
		boost::intrusive::get_parent_from_member<RootDataAccessSequence>(
			n,
			&RootDataAccessSequence::_rootDataAccessSequenceLinks
		);
}

inline RootDataAccessSequenceLinkingArtifacts::const_pointer
RootDataAccessSequenceLinkingArtifacts::to_value_ptr(RootDataAccessSequenceLinkingArtifacts::const_hook_ptr n)
{
	return (RootDataAccessSequenceLinkingArtifacts::const_pointer)
		boost::intrusive::get_parent_from_member<RootDataAccessSequence>(
			n,
			&RootDataAccessSequence::_rootDataAccessSequenceLinks
		);
}


#endif // ROOT_DATA_ACCESS_SEQUENCE_LINKING_ARTIFACTS_IMPLEMENTATION_HPP
