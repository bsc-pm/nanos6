/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>

#include "InstrumentThreadManagement.hpp"
#include "InstrumentVerbose.hpp"
#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"


using namespace Instrument::Verbose;


namespace Instrument {
	void createdThread(/* OUT */ thread_id_t &threadId, compute_place_id_t const &computePlaceId) {
		threadId = GenericIds::getNewThreadId();
		
		if (!_verboseThreadManagement) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		Instrument::InstrumentationContext tmpContext;
		tmpContext._threadId = threadId;
		logEntry->appendLocation(tmpContext);
		logEntry->_contents << " <-> CreateThread " << threadId;
		
		addLogEntry(logEntry);
	}
	
	
	void createdExternalThread_private(/* OUT */ external_thread_id_t &threadId, std::string const &name) {
		threadId = GenericIds::getNewExternalThreadId();
		
		if (!_verboseThreadManagement) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->appendLocation();
		logEntry->_contents << " <-> CreateExternalThread " << name << " " << threadId;
		
		addLogEntry(logEntry);
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
