/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#include "Message.hpp"
#include "MessageId.hpp"

#include <ClusterNode.hpp>

Message::Message(const char* name, MessageType type, size_t size, const ClusterNode *from)
{
	_deliverable = (Deliverable *)calloc(1, sizeof(msg_header) + size);
	FatalErrorHandler::failIf(
		_deliverable == nullptr,
		"Could not allocate for creating message"
	);
	
	strncpy(_deliverable->header.name, name, MSG_NAMELEN);
	_deliverable->header.type = type;
	_deliverable->header.size = size;
	/*! initialize the message id to 0 for now. In the
	 * future, it will probably be something related to
	 * the Task related with this message. */
	_deliverable->header.id = MessageId::nextMessageId();
	_deliverable->header.senderId = from->getIndex();
	
	_messengerData = nullptr;
	_delivered = false;
}
