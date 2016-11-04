#include <cassert>

#include "InstrumentTaskWait.hpp"
#include "InstrumentVerbose.hpp"
#include "executors/threads/WorkerThread.hpp"


using namespace Instrument::Verbose;


namespace Instrument {
	void enterTaskWait(task_id_t taskId, char const *invocationSource) {
		if (!_verboseTaskWait) {
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
		logEntry->_contents << " --> TaskWait " << (invocationSource ? invocationSource : "") << " task:" << taskId;
		
		addLogEntry(logEntry);
	}
	
	
	void exitTaskWait(task_id_t taskId) {
		if (!_verboseTaskWait) {
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
		logEntry->_contents << " <-- TaskWait task:" << taskId;
		
		addLogEntry(logEntry);
	}
	
	
}
