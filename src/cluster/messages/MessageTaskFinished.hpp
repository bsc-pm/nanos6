/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef MESSAGE_TASK_FINISHED_HPP
#define MESSAGE_TASK_FINISHED_HPP

#include <sstream>

#include "Message.hpp"

class MessageTaskFinished : public Message {
	struct TaskFinishedMessageContent {
		//! An opaque id that that will uniquely identifies the
		//! offloaded task
		void *_offloadedTaskId;
	};
	
	//! pointer to message payload
	TaskFinishedMessageContent *_content;
	
public:
	MessageTaskFinished(const ClusterNode *from, void *offloadedTaskId);
	
	MessageTaskFinished(Deliverable *dlv)
		: Message(dlv)
	{
		_content = reinterpret_cast<TaskFinishedMessageContent *>(_deliverable->payload);
	}
	
	bool handleMessage();
	
	inline std::string toString() const
	{
		std::stringstream ss;
		ss << "[offloadedTaskId:" << _content->_offloadedTaskId << "]";
		
		return ss.str();
	}
};

#endif /* MESSAGE_TASK_FINISHED_HPP */
