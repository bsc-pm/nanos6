/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#include "InstrumentBlocking.hpp"
#include "InstrumentVerbose.hpp"

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupport.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp>


using namespace Instrument::Verbose;



namespace Instrument {
	void enterBlocking(
		task_id_t taskId,
		InstrumentationContext const &context
	) {
		if (!_verboseBlocking) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " --> nanos_block_current_task task:" << taskId;
		
		addLogEntry(logEntry);
	}
	
	void exitBlocking(
		task_id_t taskId,
		InstrumentationContext const &context
	) {
		if (!_verboseBlocking) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-- nanos_block_current_task task:" << taskId;
		
		addLogEntry(logEntry);
	}
	
	void unblockTask(
		task_id_t taskId,
		InstrumentationContext const &context
	) {
		if (!_verboseBlocking) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> nanos_unblock_task task:" << taskId;
		
		addLogEntry(logEntry);
	}
}

