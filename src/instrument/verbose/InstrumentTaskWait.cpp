/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include "InstrumentTaskWait.hpp"
#include "InstrumentVerbose.hpp"

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupport.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp>


using namespace Instrument::Verbose;


namespace Instrument {
	void enterTaskWait(task_id_t taskId, char const *invocationSource, __attribute__((unused)) task_id_t if0TaskId, InstrumentationContext const &context) {
		if (!_verboseTaskWait) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " --> TaskWait " << (invocationSource ? invocationSource : "") << " task:" << taskId;
		
		addLogEntry(logEntry);
	}
	
	
	void exitTaskWait(task_id_t taskId, InstrumentationContext const &context) {
		if (!_verboseTaskWait) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-- TaskWait task:" << taskId;
		
		addLogEntry(logEntry);
	}
	
	
}
