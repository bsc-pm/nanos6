/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include "InstrumentUserMutex.hpp"
#include "InstrumentVerbose.hpp"

#include <InstrumentInstrumentationContext.hpp>


using namespace Instrument::Verbose;


namespace Instrument {
	void acquiredUserMutex(UserMutex *userMutex, InstrumentationContext const &context) {
		if (!_verboseUserMutex) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " --> UserMutex " << userMutex;
		
		addLogEntry(logEntry);
	}
	
	
	void blockedOnUserMutex(UserMutex *userMutex, InstrumentationContext const &context) {
		if (!_verboseUserMutex) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " --- BlockedOnUserMutex " << userMutex;
		
		addLogEntry(logEntry);
	}
	
	
	void releasedUserMutex(UserMutex *userMutex, InstrumentationContext const &context) {
		if (!_verboseUserMutex) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-- UserMutex " << userMutex;
		
		addLogEntry(logEntry);
	}

	void enterUserMutexLock(
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	void exitUserMutexLock(
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	void enterUserMutexUnlock(
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	void exitUserMutexUnlock(
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

}
