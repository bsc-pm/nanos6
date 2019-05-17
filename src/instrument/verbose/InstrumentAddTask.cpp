/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#include <atomic>
#include <cassert>

#include "InstrumentAddTask.hpp"
#include "InstrumentVerbose.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include <InstrumentInstrumentationContext.hpp>


using namespace Instrument::Verbose;


namespace Instrument {
	task_id_t enterAddTask(
		nanos6_task_info_t *taskInfo, nanos6_task_invocation_info_t *taskInvokationInfo, __attribute__((unused)) size_t flags,
		InstrumentationContext const &context
	) {
		static std::atomic<task_id_t::inner_type_t> _nextTaskId(0);
		
		task_id_t taskId = _nextTaskId++;
		
		if (_verboseAddTask) {
			LogEntry *logEntry = getLogEntry(context);
			assert(logEntry != nullptr);
			
			logEntry->appendLocation(context);
			
			logEntry->_contents << " --> AddTask " << taskId;
			if (taskInfo && taskInfo->implementations[0].task_label) {
				logEntry->_contents << " " << taskInfo->implementations[0].task_label;
			}
			if (taskInvokationInfo && taskInvokationInfo->invocation_source) {
				logEntry->_contents << " " << taskInvokationInfo->invocation_source;
			}
			
			addLogEntry(logEntry);
		}
		
		return taskId;
	}
	
	
	void createdTask(
		void *taskObject,
		task_id_t taskId,
		InstrumentationContext const &context
	) {
		if (!_verboseAddTask) {
			return;
		}
		
		Task *task = (Task *) taskObject;
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " --- AddTask: created " << taskId << " object:" << task;
		if (task->getParent() != nullptr) {
			logEntry->_contents << " parent:" << task->getParent()->getInstrumentationTaskId();
		}
		
		addLogEntry(logEntry);
	}
	
	
	void exitAddTask(
		task_id_t taskId,
		InstrumentationContext const &context
	) {
		if (!_verboseAddTask) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-- AddTask " << taskId;
		
		addLogEntry(logEntry);
	}
	
	
}
