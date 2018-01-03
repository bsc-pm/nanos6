/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupport.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp>

#include "InstrumentTaskStatus.hpp"
#include "InstrumentVerbose.hpp"


using namespace Instrument::Verbose;


namespace Instrument {
	void taskIsPending(task_id_t taskId, InstrumentationContext const &context) {
		if (!_verboseTaskStatus) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> TaskStatusChange " << taskId << " to:pending";
		
		addLogEntry(logEntry);
	}
	
	
	void taskIsReady(task_id_t taskId, InstrumentationContext const &context) {
		if (!_verboseTaskStatus) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> TaskStatusChange " << taskId << " to:ready";
		
		addLogEntry(logEntry);
	}
	
	
	void taskIsExecuting(task_id_t taskId, InstrumentationContext const &context) {
		if (!_verboseTaskStatus) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> TaskStatusChange " << taskId << " to:executing";
		
		addLogEntry(logEntry);
	}
	
	
	void taskIsBlocked(task_id_t taskId, task_blocking_reason_t reason, InstrumentationContext const &context) {
		if (!_verboseTaskStatus) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> TaskStatusChange " << taskId << " to:blocked";
		logEntry->_contents << " reason:";
		switch (reason) {
			case in_taskwait_blocking_reason:
				logEntry->_contents << "taskwait";
				break;
			case in_mutex_blocking_reason:
				logEntry->_contents << "mutex";
				break;
			default:
				logEntry->_contents << "unknown";
				break;
		}
		
		addLogEntry(logEntry);
	}
	
	
	void taskIsZombie(task_id_t taskId, InstrumentationContext const &context) {
		if (!_verboseTaskStatus) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> TaskStatusChange " << taskId << " to:zombie";
		
		addLogEntry(logEntry);
	}
	
	
}
