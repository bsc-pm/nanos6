/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <atomic>
#include <cassert>

#include "InstrumentAddTask.hpp"
#include "InstrumentVerbose.hpp"
#include "tasks/Task.hpp"

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupport.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp>


using namespace Instrument::Verbose;


namespace Instrument {
	task_id_t enterAddTask(
		nanos_task_info *taskInfo, nanos_task_invocation_info *taskInvokationInfo, __attribute__((unused)) size_t flags,
		InstrumentationContext const &context
	) {
		static std::atomic<task_id_t::inner_type_t> _nextTaskId(0);
		
		if (!_verboseAddTask) {
			return task_id_t();
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		task_id_t taskId = _nextTaskId++;
		logEntry->appendLocation(context);
		
		logEntry->_contents << " --> AddTask " << taskId;
		if (taskInfo && taskInfo->task_label) {
			logEntry->_contents << " " << taskInfo->task_label;
		}
		if (taskInvokationInfo && taskInvokationInfo->invocation_source) {
			logEntry->_contents << " " << taskInvokationInfo->invocation_source;
		}
		
		addLogEntry(logEntry);
		
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
		
		LogEntry *logEntry = getLogEntry();
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
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-- AddTask " << taskId;
		
		addLogEntry(logEntry);
	}
	
	
}
