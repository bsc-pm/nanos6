#include <cassert>

#include "InstrumentThreadManagement.hpp"
#include "InstrumentVerbose.hpp"
#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"

#include "../generic_ids/GenericIds.hpp"


using namespace Instrument::Verbose;


namespace Instrument {
	thread_id_t createdThread() {
		thread_id_t::inner_type_t threadId = GenericIds::getNewThreadId();
		
		if (!_verboseThreadManagement) {
			return thread_id_t(threadId);
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->appendLocation();
		logEntry->_contents << " <-> CreateThread " << threadId;
		
		addLogEntry(logEntry);
		
		return thread_id_t(threadId);
	}
	
	
	void threadWillSuspend(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t computePlaceID) {
		if (!_verboseThreadManagement) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->appendLocation();
		logEntry->_contents << " --> SuspendThread ";
		
		addLogEntry(logEntry);
	}
	
	
	void threadHasResumed(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t computePlaceID) {
		if (!_verboseThreadManagement) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->appendLocation();
		logEntry->_contents << " <-- SuspendThread ";
		
		addLogEntry(logEntry);
	}
	
	
}
