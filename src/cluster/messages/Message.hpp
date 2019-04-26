/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef MESSAGE_HPP
#define MESSAGE_HPP

#include "MessageType.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "lowlevel/threads/KernelLevelThread.hpp"
#include "support/GenericFactory.hpp"

class ClusterNode;

class Message {
public:
	struct msg_header {
		//! string containing the message type
		char name[MSG_NAMELEN];
		
		//! the type of the message
		MessageType type;
		
		//! size of the payload in bytes
		size_t size;
		
		//! Id of the message
		int id;
		
		//! Cluster index of the sender node
		int senderId;
	};
	
	//! Deliverable is the structure that is actually sent over the network.
	//!
	//! It contains a message header and a payload that is MessageType specific.
	//! This struct is sent as is over the network without any serialisation.
	typedef struct {
		struct msg_header header;
		char payload[];
	} Deliverable;
	
private:
	//! An opaque pointer to Messenger-specific data
	void * _messengerData;
	
	//! Flag indicating whether the Message has been delivered
	bool _delivered;
	
protected:
	//! The part of the message we actually send over the network
	Deliverable *_deliverable;
	
public:
	Message() = delete;
	Message(const char* name, MessageType type, size_t size, const ClusterNode *from);
	
	//! Construct a message from a received(?) Deliverable structure
	Message(Deliverable *dlv)
		: _messengerData(nullptr), _delivered(false),
		_deliverable(dlv)
	{
		assert(dlv != nullptr);
	}
	
	virtual ~Message()
	{
		assert(_deliverable != nullptr);
		free(_deliverable);
	}
	
	//! \brief Returns the type of the Message
	inline MessageType getType() const
	{
		return _deliverable->header.type;
	}
	
	//! \brief Returns the size of the Message
	inline size_t getSize() const
	{
		return _deliverable->header.type;
	}
	
	//! \brief Returns the id of the Message
	inline int getId() const
	{
		return _deliverable->header.id;
	}
	
	//! \brief Returns the cluster id of the node that sent the  Message
	inline int getSenderId() const
	{
		return _deliverable->header.senderId;
	}
	
	//! \brief Return the Deliverable data of the Message
	inline Deliverable *getDeliverable() const
	{
		return _deliverable;
	}
	
	//! \brief Return the Messenger-specific data
	inline void *getMessengerData() const
	{
		return _messengerData;
	}
	
	//! \brief Set the Messenger-specific data
	inline void setMessengerData(void *data)
	{
		_messengerData = data;
	}
	
	//! \brief Mark the Message as delivered
	inline void markAsDelivered()
	{
		_delivered = true;
	}
	
	//! \brief Check if the Message is delivered
	inline bool isDelivered() const
	{
		return _delivered;
	}
	
	//! \brief Handles the received message
	//!
	//! Specific to each type of message. This method handles the
	//! MessageType specific operation. It returns true if the Message
	//! can be delete or false otherwise.
	virtual bool handleMessage() = 0;
	
	//! \brief prints info about the message
	virtual void toString(std::ostream& where) const = 0;
	
	friend std::ostream& operator<<(std::ostream& out, const Message& msg)
	{
		msg.toString(out);
		return out;
	}
};

#define REGISTER_MSG_CLASS(NAME, CREATEFN) \
	GenericFactory<int, Message*, Message::Deliverable*>::getInstance().emplace(NAME, CREATEFN)

#endif /* MESSAGE_HPP */
