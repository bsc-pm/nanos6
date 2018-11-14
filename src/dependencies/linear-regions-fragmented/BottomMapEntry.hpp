/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef BOTTOM_MAP_ENTRY_HPP
#define BOTTOM_MAP_ENTRY_HPP


#include <boost/intrusive/avl_set.hpp>
#include <boost/intrusive/avl_set_hook.hpp>

#include "DataAccessLink.hpp"
#include "DataAccessRegion.hpp"
#include "../DataAccessType.hpp"
#include "ReductionSpecific.hpp"


struct BottomMapEntry;


struct BottomMapEntryLinkingArtifacts {
	#if NDEBUG
		typedef boost::intrusive::link_mode<boost::intrusive::normal_link> link_mode_t;
	#else
		typedef boost::intrusive::link_mode<boost::intrusive::safe_link> link_mode_t;
	#endif
	
	typedef boost::intrusive::avl_set_member_hook<link_mode_t> hook_type;
	typedef hook_type* hook_ptr;
	typedef const hook_type* const_hook_ptr;
	typedef BottomMapEntry value_type;
	typedef value_type* pointer;
	typedef const value_type* const_pointer;
	
	static inline constexpr hook_ptr to_hook_ptr (value_type &value);
	static inline constexpr const_hook_ptr to_hook_ptr(const value_type &value);
	static inline pointer to_value_ptr(hook_ptr n);
	static inline const_pointer to_value_ptr(const_hook_ptr n);
};


struct BottomMapEntryContents {
	DataAccessLink _link;
	DataAccessType _accessType;
	reduction_type_and_operator_index_t _reductionTypeAndOperatorIndex;
	
	BottomMapEntryContents(DataAccessLink link, DataAccessType accessType,
		reduction_type_and_operator_index_t reductionTypeAndOperatorIndex)
		: _link(link), _accessType(accessType),
		_reductionTypeAndOperatorIndex(reductionTypeAndOperatorIndex)
	{
	}
};


struct BottomMapEntry : public BottomMapEntryContents {
	BottomMapEntryLinkingArtifacts::hook_type _links;
	
	DataAccessRegion _region;
	
	BottomMapEntry(DataAccessRegion region, DataAccessLink link, DataAccessType accessType,
		reduction_type_and_operator_index_t reductionTypeAndOperatorIndex)
		: BottomMapEntryContents(link, accessType, reductionTypeAndOperatorIndex),
		_links(), _region(region)
	{
	}
	
	DataAccessRegion const &getAccessRegion() const
	{
		return _region;
	}
	
	void setAccessRegion(DataAccessRegion const &newRegion)
	{
		_region = newRegion;
	}
	
};


inline constexpr BottomMapEntryLinkingArtifacts::hook_ptr
BottomMapEntryLinkingArtifacts::to_hook_ptr (BottomMapEntryLinkingArtifacts::value_type &value)
{
	return &value._links;
}

inline constexpr BottomMapEntryLinkingArtifacts::const_hook_ptr
BottomMapEntryLinkingArtifacts::to_hook_ptr(const BottomMapEntryLinkingArtifacts::value_type &value)
{
	return &value._links;
}

inline BottomMapEntryLinkingArtifacts::pointer
BottomMapEntryLinkingArtifacts::to_value_ptr(BottomMapEntryLinkingArtifacts::hook_ptr n)
{
	return (BottomMapEntryLinkingArtifacts::pointer)
		boost::intrusive::get_parent_from_member<BottomMapEntry>(
			n,
			&BottomMapEntry::_links
		);
}

inline BottomMapEntryLinkingArtifacts::const_pointer
BottomMapEntryLinkingArtifacts::to_value_ptr(BottomMapEntryLinkingArtifacts::const_hook_ptr n)
{
	return (BottomMapEntryLinkingArtifacts::const_pointer)
		boost::intrusive::get_parent_from_member<BottomMapEntry>(
			n,
			&BottomMapEntry::_links
		);
}


#endif // BOTTOM_MAP_ENTRY_HPP
