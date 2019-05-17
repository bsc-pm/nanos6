/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include <InstrumentInstrumentationContext.hpp>

#include "InstrumentDependenciesByAccess.hpp"
#include "InstrumentVerbose.hpp"


using namespace Instrument::Verbose;


namespace Instrument {
	void registerTaskAccess(
		task_id_t taskId, DataAccessType accessType, bool weak, void *start, size_t length,
		InstrumentationContext const &context
	) {
		if (!_verboseDependenciesByAccess) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> RegisterTaskAccess Task:" << taskId << " ";
		
		if (weak) {
			logEntry->_contents << "weak ";
		}
		switch (accessType) {
			case READ_ACCESS_TYPE:
				logEntry->_contents << "input";
				break;
			case READWRITE_ACCESS_TYPE:
				logEntry->_contents << "inout";
				break;
			case WRITE_ACCESS_TYPE:
				logEntry->_contents << "output";
				break;
			default:
				logEntry->_contents << "unknown_access_type";
				break;
		}
		
		logEntry->_contents << " start:" << start << " bytes:" << length;
		
		addLogEntry(logEntry);
	}
	
	
}
