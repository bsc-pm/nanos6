#include <cassert>

#include "InstrumentThreadManagement.hpp"
#include "InstrumentVerbose.hpp"
#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"


using namespace Instrument::Verbose;


namespace Instrument {
	void createdThread(WorkerThread *thread) {
		if (!_verboseThreadManagement) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
		
		if (currentWorker != nullptr) {
			logEntry->_contents << "Thread:" << currentWorker << " CPU:" << currentWorker->getCpuId();
		} else {
			logEntry->_contents << "Thread:LeaderThread CPU:ANY";
		}
		
		logEntry->_contents << " <-> CreateThread " << thread;
		
		addLogEntry(logEntry);
	}
	
	
	void threadWillSuspend(WorkerThread *thread, CPU *cpu) {
		if (!_verboseThreadManagement) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->_contents << "Thread:" << thread << " CPU:" << cpu->_virtualCPUId << " --> SuspendThread ";
		
		addLogEntry(logEntry);
	}
	
	
	void threadHasResumed(WorkerThread *thread, CPU *cpu) {
		if (!_verboseThreadManagement) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->_contents << "Thread:" << thread << " CPU:" << cpu->_virtualCPUId << " <-- SuspendThread ";
		
		addLogEntry(logEntry);
	}
	
	
}
