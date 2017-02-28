#ifndef CACHE_TRACKING_OBJECT_LINKING_ARTIFACTS_IMPLEMENTATION_HPP
#define CACHE_TRACKING_OBJECT_LINKING_ARTIFACTS_IMPLEMENTATION_HPP

#include <boost/intrusive/parent_from_member.hpp>

#include "CacheTrackingObject.hpp"
#include "CacheTrackingObjectLinkingArtifacts.hpp"


inline constexpr CacheTrackingObjectLinkingArtifacts::hook_ptr
CacheTrackingObjectLinkingArtifacts::to_hook_ptr (CacheTrackingObjectLinkingArtifacts::value_type &value)
{
	return &value._CacheTrackingObjectLinks._objectsHook;
}

inline constexpr CacheTrackingObjectLinkingArtifacts::const_hook_ptr
CacheTrackingObjectLinkingArtifacts::to_hook_ptr(const CacheTrackingObjectLinkingArtifacts::value_type &value)
{
	return &value._CacheTrackingObjectLinks._objectsHook;
}

inline CacheTrackingObjectLinkingArtifacts::pointer
CacheTrackingObjectLinkingArtifacts::to_value_ptr(CacheTrackingObjectLinkingArtifacts::hook_ptr n)
{
	return (CacheTrackingObjectLinkingArtifacts::pointer)
		boost::intrusive::get_parent_from_member<CacheTrackingObject>(
			boost::intrusive::get_parent_from_member<CacheTrackingObjectHooks>(
				n,
				&CacheTrackingObjectHooks::_objectsHook
			),
			&CacheTrackingObject::_CacheTrackingObjectLinks
		);
}

inline CacheTrackingObjectLinkingArtifacts::const_pointer
CacheTrackingObjectLinkingArtifacts::to_value_ptr(CacheTrackingObjectLinkingArtifacts::const_hook_ptr n)
{
	return (CacheTrackingObjectLinkingArtifacts::const_pointer)
		boost::intrusive::get_parent_from_member<CacheTrackingObject>(
			boost::intrusive::get_parent_from_member<CacheTrackingObjectHooks>(
				n,
				&CacheTrackingObjectHooks::_objectsHook
			),
			&CacheTrackingObject::_CacheTrackingObjectLinks
		);
}


#endif // CACHE_TRACKING_OBJECT_LINKING_ARTIFACTS_IMPLEMENTATION_HPP
