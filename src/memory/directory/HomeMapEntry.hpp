/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef HOME_MAP_ENTRY_HPP
#define HOME_MAP_ENTRY_HPP


#include <boost/intrusive/avl_set.hpp>
#include <boost/intrusive/avl_set_hook.hpp>

#include "DataAccessRegion.hpp"

class Task;
class MemoryPlace;

struct HomeMapEntry;

struct HomeMapEntryLinkingArtifacts {
	#if NDEBUG
	typedef boost::intrusive::link_mode<boost::intrusive::normal_link> link_mode_t;
	#else
	typedef boost::intrusive::link_mode<boost::intrusive::safe_link> link_mode_t;
	#endif
	
	typedef boost::intrusive::avl_set_member_hook<link_mode_t> hook_type;
	typedef hook_type* hook_ptr;
	typedef const hook_type* const_hook_ptr;
	typedef HomeMapEntry value_type;
	typedef value_type* pointer;
	typedef const value_type* const_pointer;
	
	static inline constexpr hook_ptr to_hook_ptr(value_type &value);
	static inline constexpr const_hook_ptr to_hook_ptr(const value_type &value);
	static inline pointer to_value_ptr(hook_ptr n);
	static inline const_pointer to_value_ptr(const_hook_ptr n);
};


class HomeMapEntry {
	HomeMapEntryLinkingArtifacts::hook_type _links;
	
	//! \brief The region we are mapping
	DataAccessRegion _region;
	
	//! \brief The home node of the region
	MemoryPlace const *_homeNode;
	
public:
	
	HomeMapEntry(DataAccessRegion region, MemoryPlace const *homeNode)
		: _links(), _region(region), _homeNode(homeNode)
	{
		assert(homeNode != nullptr);
	}
	
	//! \brief Get the access region of the mapping
	inline DataAccessRegion const &getAccessRegion() const
	{
		return _region;
	}
	
	//! \brief Set the region of the mapping
	inline void setAccessRegion(DataAccessRegion const &newRegion)
	{
		_region = newRegion;
	}
	
	//! \brief Get the home node of the region
	inline MemoryPlace const *getHomeNode() const
	{
		return _homeNode;
	}
	
	//! \brief Set the home node of the region
	inline void setHomeNode(MemoryPlace const *homeNode)
	{
		_homeNode = homeNode;
	}
	
	friend class HomeMapEntryLinkingArtifacts;
};


inline constexpr HomeMapEntryLinkingArtifacts::hook_ptr
HomeMapEntryLinkingArtifacts::to_hook_ptr(HomeMapEntryLinkingArtifacts::value_type &value)
{
	return &value._links;
}

inline constexpr HomeMapEntryLinkingArtifacts::const_hook_ptr
HomeMapEntryLinkingArtifacts::to_hook_ptr(const HomeMapEntryLinkingArtifacts::value_type &value)
{
	return &value._links;
}

inline HomeMapEntryLinkingArtifacts::pointer
HomeMapEntryLinkingArtifacts::to_value_ptr(HomeMapEntryLinkingArtifacts::hook_ptr n)
{
	return (HomeMapEntryLinkingArtifacts::pointer)
		boost::intrusive::get_parent_from_member<HomeMapEntry>(
			n,
			&HomeMapEntry::_links
		);
}

inline HomeMapEntryLinkingArtifacts::const_pointer
HomeMapEntryLinkingArtifacts::to_value_ptr(HomeMapEntryLinkingArtifacts::const_hook_ptr n)
{
	return (HomeMapEntryLinkingArtifacts::const_pointer)
		boost::intrusive::get_parent_from_member<HomeMapEntry>(
			n,
			&HomeMapEntry::_links
		);
}

#endif // HOME_MAP_ENTRY_HPP
