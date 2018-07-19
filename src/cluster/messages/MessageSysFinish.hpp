/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef __MESSAGE_SYS_FINISH_HPP__
#define __MESSAGE_SYS_FINISH_HPP__

#include "Message.hpp"

class MessageSysFinish : public Message
{
public:
	MessageSysFinish(const ClusterNode *from);
	MessageSysFinish(Deliverable *dlv);
	void handleMessage();
	void toString(std::ostream &where) const;
};

MessageSysFinish::MessageSysFinish(Deliverable *dlv)
	: Message(dlv)
{
}

inline void MessageSysFinish::toString(std::ostream &where) const
{
	where << "SysFinish";
}

//! Register the Message type to the Object factory
namespace {
	Message *createSysFinishMessage(Message::Deliverable *dlv)
	{
		return new MessageSysFinish(dlv);
	}
	
	const bool __attribute__((unused))_registered_sys_finish =
		REGISTER_MSG_CLASS(SYS_FINISH, createSysFinishMessage);
}

#endif /* __MESSAGE_SYS_FINISH_HPP__ */
