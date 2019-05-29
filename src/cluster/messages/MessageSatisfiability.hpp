/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef MESSAGE_SATISFIABILITY_HPP
#define MESSAGE_SATISFIABILITY_HPP

#include "Message.hpp"

#include <SatisfiabilityInfo.hpp>

class MessageSatisfiability : public Message {
	struct SatisfiabilityMessageContent {
		//! The opaque id identifying the offloaded task
		void *_offloadedTaskId;
		
		//! Satisfiability information we are sending
		TaskOffloading::SatisfiabilityInfo _satInfo;
	};
	
	//! pointer to message payload
	SatisfiabilityMessageContent *_content;
	
public:
	MessageSatisfiability(const ClusterNode *from, void *offloadedTaskId,
			TaskOffloading::SatisfiabilityInfo const &satInfo);
	
	MessageSatisfiability(Deliverable *dlv)
		: Message(dlv)
	{
		_content = reinterpret_cast<SatisfiabilityMessageContent *>(_deliverable->payload);
	}
	
	bool handleMessage();
	
	inline void toString(std::ostream &where) const
	{
		where << "SatInfo " << _content->_satInfo;
	}
};

#endif /* MESSAGE_SATISFIABILITY_HPP */
