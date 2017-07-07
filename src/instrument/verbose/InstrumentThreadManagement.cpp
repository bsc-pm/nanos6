#include <cassert>

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupport.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp>

#include "InstrumentThreadManagement.hpp"
#include "InstrumentVerbose.hpp"
#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"


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
	
	void threadWillShutdown() {
		if (!_verboseThreadManagement) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->appendLocation();
		logEntry->_contents << " <-> ShutdownThread ";
		
		addLogEntry(logEntry);
	}
	
	void threadEnterBusyWait(busy_wait_reason_t reason)
	{
		if (!_verboseThreadManagement) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->appendLocation();
		logEntry->_contents << " --> BusyWait ";
		switch (reason) {
			case scheduling_polling_slot_busy_wait_reason:
				logEntry->_contents << "(scheduler polling) ";
				break;
		}
		
		addLogEntry(logEntry);
	}
	
	void threadExitBusyWait()
	{
		if (!_verboseThreadManagement) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->appendLocation();
		logEntry->_contents << " <-- BusyWait ";
		
		addLogEntry(logEntry);
	}
}
