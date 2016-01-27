#include <cassert>

#include "InstrumentDependenciesByAccess.hpp"
#include "InstrumentVerbose.hpp"
#include "executors/threads/WorkerThread.hpp"


using namespace Instrument::Verbose;


namespace Instrument {
	void registerTaskAccess(task_id_t taskId, DataAccessType accessType, bool weak, void *start, size_t length) {
		if (!_verboseDependenciesByAccess) {
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
