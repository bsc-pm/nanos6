#include <cassert>

#include "InstrumentUserMutex.hpp"
#include "InstrumentVerbose.hpp"
#include "executors/threads/WorkerThread.hpp"


using namespace Instrument::Verbose;


namespace Instrument {
	void acquiredUserMutex(UserMutex *userMutex) {
		if (!_verboseUserMutex) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
		
		if (currentWorker != nullptr) {
			logEntry->_contents << "Thread:" << currentWorker << " CPU:" << currentWorker->getCpuId();
		} else {
			logEntry->_contents << "Thread:external CPU:ANY";
		}
		logEntry->_contents << " --> UserMutex " << userMutex;
		
		addLogEntry(logEntry);
	}
	
	
	void blockedOnUserMutex(UserMutex *userMutex) {
		if (!_verboseUserMutex) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
		
		if (currentWorker != nullptr) {
			logEntry->_contents << "Thread:" << currentWorker << " CPU:" << currentWorker->getCpuId();
		} else {
			logEntry->_contents << "Thread:external CPU:ANY";
		}
		logEntry->_contents << " --- BlockedOnUserMutex " << userMutex;
		
		addLogEntry(logEntry);
	}
	
	
	void releasedUserMutex(UserMutex *userMutex) {
		if (!_verboseUserMutex) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
		
		if (currentWorker != nullptr) {
			logEntry->_contents << "Thread:" << currentWorker << " CPU:" << currentWorker->getCpuId();
		} else {
			logEntry->_contents << "Thread:external CPU:ANY";
		}
		logEntry->_contents << " <-- UserMutex " << userMutex;
		
		addLogEntry(logEntry);
	}
	
	
}
