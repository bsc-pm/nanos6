/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef MESSAGE_RELEASE_ACCESS_HPP
#define MESSAGE_RELEASE_ACCESS_HPP

#include "Message.hpp"
#include "dependencies/DataAccessType.hpp"

#include <DataAccessRegion.hpp>

class MessageReleaseAccess : public Message {
	struct ReleaseAccessMessageContent {
		//! The opaque id identifying the offloaded task
		void *_offloadedTaskId;
		
		//! The region we are releasing
		DataAccessRegion _region;
		
		//! The type of the access
		DataAccessType _type;
		
		//! true if access is weak
		bool _weak;
		
		//! The location on which the access is being released
		int _location;
	};
	
	//! pointer to message payload
	ReleaseAccessMessageContent *_content;
	
public:
	MessageReleaseAccess(const ClusterNode *from, void *offloadedTaskId,
			DataAccessRegion const &region, DataAccessType type,
			bool weak, int location);
	
	MessageReleaseAccess(Deliverable *dlv)
		: Message(dlv)
	{
		_content = reinterpret_cast<ReleaseAccessMessageContent *>(_deliverable->payload);
	}
	
	bool handleMessage();
	
	inline void toString(std::ostream &where) const
	{
		where << "ReleaseAccess region:" << _content->_region
			<< " location:" << _content->_location;
	}
};

#endif /* MESSAGE_RELEASE_ACCESS_HPP */
