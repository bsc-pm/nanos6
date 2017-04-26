#include <cassert>

#include "InstrumentTaskExecution.hpp"
#include "InstrumentVerbose.hpp"
#include "executors/threads/CPU.hpp"

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupport.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp>


using namespace Instrument::Verbose;


namespace Instrument {
	void startTask(task_id_t taskId, InstrumentationContext const &context) {
		if (!_verboseTaskExecution) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " --> Task " << taskId;
		
		addLogEntry(logEntry);
	}
	
	
	void returnToTask(task_id_t taskId, InstrumentationContext const &context) {
		if (!_verboseTaskExecution) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " ->> Task " << taskId;
		
		addLogEntry(logEntry);
	}
	
	
	void endTask(task_id_t taskId, InstrumentationContext const &context) {
		if (!_verboseTaskExecution) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-- Task " << taskId;
		
		addLogEntry(logEntry);
	}
	
	
	void destroyTask(task_id_t taskId, InstrumentationContext const &context) {
		if (!_verboseTaskExecution) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->appendLocation(context);
		logEntry->_contents << " <-> DestroyTask " << taskId;
		
		addLogEntry(logEntry);
	}
	
}
