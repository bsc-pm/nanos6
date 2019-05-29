/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef MESSAGE_TASKNEW_HPP
#define MESSAGE_TASKNEW_HPP

#include "Message.hpp"

#include <nanos6/task-instantiation.h>
#include <SatisfiabilityInfo.hpp>

class MessageTaskNew : public Message {
	struct TaskNewMessageContent {
		//! Necessary info for duplicating the task on the remote node
		nanos6_task_info_t _taskInfo;
		nanos6_task_invocation_info_t _taskInvocationInfo;
		
		//! The flags of the task
		size_t _flags;
		
		//! The size of the Task's argsBlock
		size_t _argsBlockSize;
		
		//! The number of task implementations
		size_t _numImplementations;
		
		//! The number of satisfiability information entries
		size_t _numSatInfo;
		
		//! An opaque id that that will uniquely identifies the
		//! offloaded task
		void *_offloadedTaskId;
		
		//! buffer holding all the variable length information we need
		//! to send across.
		//!
		//! This includes the actual argsBlock of the task the task
		//! implementation information and satisfiability information.
		//!
		//! The format looks like this:
		//!
		//! [nanos6_task_implementation_info_t | ... | SatisfiabilityInfo | ... | argsBlock]
		//!
		//! If you need to change this layout amend the previous.
		char _msgData[];
	};
	
	//! pointer to message payload
	TaskNewMessageContent *_content;
	
	//! Returns a pointer in the Message memory holding the task
	//! implementation info
	inline nanos6_task_implementation_info_t *getImplementationsPtr() const
	{
		return (nanos6_task_implementation_info_t *)_content->_msgData;
	}
	
	//! Returns a pointer in the Message memory holding the satisfiability
	//! information we have
	inline TaskOffloading::SatisfiabilityInfo *getSatInfoPtr() const
	{
		return (TaskOffloading::SatisfiabilityInfo *)
			(getImplementationsPtr() + _content->_numImplementations);
	}
	
	//! Returns a pointer in the Message memory holding the argsBlock
	inline void *getArgsBlockPtr() const
	{
		return (void *)
			(getSatInfoPtr() + _content->_numSatInfo);
	}
	
public:
	MessageTaskNew(const ClusterNode *from, nanos6_task_info_t *taskInfo,
		nanos6_task_invocation_info_t *taskInvokationInfo, size_t flags,
		size_t numImplementations,
		nanos6_task_implementation_info_t *taskImplementations,
		size_t numSatInfo, TaskOffloading::SatisfiabilityInfo const *satInfo,
		size_t argsBlockSize, void *argsBlock,
		void *offloadedTaskId);
	
	MessageTaskNew(Deliverable *dlv)
		: Message(dlv)
	{
		_content = reinterpret_cast<TaskNewMessageContent *>(_deliverable->payload);
	}
	
	//! Get the task_info_t of the offloaded task
	//!
	//! This returns a pointer in memory of the Message. The calling site
	//! is responsible for copying out from this memory or making sure that
	//! the memory of the MessageTaskNew is not deallocated for as long as
	//! needed.
	inline nanos6_task_info_t *getTaskInfo() const
	{
		return &_content->_taskInfo;
	}
	
	//! Get the task_invocation_info_t of the offloaded task
	//!
	//! This returns a pointer in memory of the Message. The calling site
	//! is responsible for copying out from this memory or making sure that
	//! the memory of the MessageTaskNew is not deallocated for as long as
	//! needed
	inline nanos6_task_invocation_info_t *getTaskInvocationInfo() const
	{
		return &_content->_taskInvocationInfo;
	}
	
	//! Get the task flags
	inline size_t getFlags() const
	{
		return _content->_flags;
	}
	
	//! Get the task id of the offloaded Task
	inline void *getOffloadedTaskId() const
	{
		return _content->_offloadedTaskId;
	}
	
	//! Get an array of the available task implementations
	//!
	//! This returns a pointer in Message memory holding the information of
	//! the Task's implementations. The calling site is responsible for
	//! copying out from this memory or making sure that the Message is not
	//! deleted for as long as it is needed. The method also returns the
	//! number of the available task implementations.
	inline nanos6_task_implementation_info_t *getImplementations(
		size_t &numImplementations) const
	{
		numImplementations = _content->_numImplementations;
		if (numImplementations == 0) {
			return nullptr;
		}
		
		return getImplementationsPtr();
	}
	
	//! Get an array of the available satisfiability information we have
	//!
	//! This returns a pointer in memory of the Message. The calling site
	//! is responsible for copying out from this memory or making sure that
	//! the memory of the MessageTaskNew is not deallocated for as long as
	//! it is needed. The method also returns the number of available
	//! SatisfiabilityInfo structs included in the Message.
	inline TaskOffloading::SatisfiabilityInfo *getSatisfiabilityInfo(
		size_t &numSatInfo) const
	{
		numSatInfo = _content->_numSatInfo;
		if (numSatInfo == 0) {
			return nullptr;
		}
		
		return getSatInfoPtr();
	}
	
	//! Get a pointer to the Task's argsBlock
	//!
	//! This returns a pointer in memory of the Message. The calling site
	//! is responsible for copying out from this memory or making sure that
	//! the memory of the MessageTaskNew is not deallocated for as long as
	//! it is needed. The method also returns the size (in bytes) of the
	//! argsBlock
	inline void *getArgsBlock(size_t &argsBlockSize) const
	{
		argsBlockSize = _content->_argsBlockSize;
		if (argsBlockSize == 0) {
			return nullptr;
		}
		
		return getArgsBlockPtr();
	}
	
	bool handleMessage();
	
	inline void toString(std::ostream &where) const
	{
		where << "TaskNew offloaded from Node:" << getSenderId();
	}
};

#endif /* MESSAGE_TASKNEW_HPP */
