/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef MESSAGE_DATA_FETCH_HPP
#define MESSAGE_DATA_FETCH_HPP

#include "Message.hpp"

#include <DataAccessRegion.hpp>

class MessageDataFetch : public Message {
	struct DataFetchMessageContent {
		//! The remote region we bring data from
		DataAccessRegion _remoteRegion;
	};
	
	//! \brief pointer to the message payload
	DataFetchMessageContent *_content;
	
public:
	MessageDataFetch(const ClusterNode *from,
		DataAccessRegion const &remoteRegion);
	
	MessageDataFetch(Deliverable *dlv)
		: Message(dlv)
	{
		_content = reinterpret_cast<DataFetchMessageContent *>(_deliverable->payload);
	}
	
	bool handleMessage();
	
	inline void toString(std::ostream &where) const
	{
		where << "DataFetch of region:" << _content->_remoteRegion <<
			" from Node:" << getSenderId();
	}
};

#endif /* MESSAGE_DATA_FETCH_HPP */
