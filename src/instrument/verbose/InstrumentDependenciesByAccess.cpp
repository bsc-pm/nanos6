#include <cassert>

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupport.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp>

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
		
		LogEntry *logEntry = getLogEntry();
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
