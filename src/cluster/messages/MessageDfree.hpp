/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef MESSAGE_DFREE_HPP
#define MESSAGE_DFREE_HPP

#include "Message.hpp"
#include <DataAccessRegion.hpp>

class MessageDfree : public Message {
	struct DfreeMessageContent {
		//! address of the distributed allocation
		void *_address;
		
		//! size in bytes of the allocation
		size_t _size;
	};
	
	//! \brief pointer to the message payload
	DfreeMessageContent *_content;
	
public:
	MessageDfree(const ClusterNode *from);
	
	MessageDfree(Deliverable *dlv)
		: Message(dlv)
	{
		_content = reinterpret_cast<DfreeMessageContent *>(_deliverable->payload);
	}
	
	bool handleMessage();
	
	//! \brief Set the address of the allocation
	inline void setAddress(void *address)
	{
		_content->_address = address;
	}
	
	//! \brief Get the address of the allocation
	inline void *getAddress() const
	{
		return _content->_address;
	}
	
	//! \brief Set the size of the allocation
	inline void setSize(size_t size)
	{
		_content->_size = size;
	}
	
	//! \brief Get the size of the allocation
	inline size_t getSize() const
	{
		return _content->_size;
	}
	
	//! \brief Write to a stream a description of the Message
	inline void toString(std::ostream &where) const
	{
		DataAccessRegion region(_content->_address, _content->_size);
		where << "Distributed deallocation of " << region;
	}
};

#endif /* MESSAGE_DFREE_HPP */
