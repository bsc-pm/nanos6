/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "MessageTaskFinished.hpp"
#include "tasks/Task.hpp"

MessageTaskFinished::MessageTaskFinished(const ClusterNode *from,
		void *offloadedTaskId)
	: Message("MessageTaskFinished", TASK_FINISHED,
		sizeof(TaskFinishedMessageContent), from)
{
	_content = reinterpret_cast<TaskFinishedMessageContent *>(_deliverable->payload);
	_content->_offloadedTaskId = offloadedTaskId;
}

bool MessageTaskFinished::handleMessage()
{
	Task *task = (Task *)_content->_offloadedTaskId;
	ExecutionWorkflow::Step *step = task->getExecutionStep();
	assert(step != nullptr);
	
	task->setExecutionStep(nullptr);
	step->releaseSuccessors();
	delete step;
	
	return true;
}

//! Register the Message type to the Object factory
static Message *createTaskFinishedMessage(Message::Deliverable *dlv)
{
	return new MessageTaskFinished(dlv);
}

static const bool __attribute__((unused))_registered_taskfinished =
	REGISTER_MSG_CLASS(TASK_FINISHED, createTaskFinishedMessage);
