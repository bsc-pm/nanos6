#include <cassert>

#include "InstrumentDependenciesByGroup.hpp"
#include "InstrumentVerbose.hpp"
#include "executors/threads/WorkerThread.hpp"


using namespace Instrument::Verbose;


namespace Instrument {
	void beginAccessGroup(__attribute__((unused)) task_id_t parentTaskId, __attribute__((unused)) void *handler, __attribute__((unused)) bool sequenceIsEmpty) {
// 		LogEntry *logEntry = getLogEntry();
// 		assert(logEntry != nullptr);
// 		
// 		WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
// 		
// 		if (currentWorker != nullptr) {
// 			logEntry->_contents << "Thread:" << currentWorker << " CPU:" << currentWorker->getCpuId();
// 		} else {
// 			logEntry->_contents << "Thread:LeaderThread CPU:ANY";
// 		}
// 		logEntry->_contents << " ";
// 		
// 		addLogEntry(logEntry);
	}


	void addTaskToAccessGroup(__attribute__((unused)) void *handler, __attribute__((unused)) task_id_t taskId) {
// 		LogEntry *logEntry = getLogEntry();
// 		assert(logEntry != nullptr);
// 		
// 		WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
// 		
// 		if (currentWorker != nullptr) {
// 			logEntry->_contents << "Thread:" << currentWorker << " CPU:" << currentWorker->getCpuId();
// 		} else {
// 			logEntry->_contents << "Thread:LeaderThread CPU:ANY";
// 		}
// 		logEntry->_contents << " ";
// 		
// 		addLogEntry(logEntry);
	}


	void removeTaskFromAccessGroup(__attribute__((unused)) void *handler, __attribute__((unused)) task_id_t taskId) {
// 		LogEntry *logEntry = getLogEntry();
// 		assert(logEntry != nullptr);
// 		
// 		WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
// 		
// 		if (currentWorker != nullptr) {
// 			logEntry->_contents << "Thread:" << currentWorker << " CPU:" << currentWorker->getCpuId();
// 		} else {
// 			logEntry->_contents << "Thread:LeaderThread CPU:ANY";
// 		}
// 		logEntry->_contents << " ";
// 		
// 		addLogEntry(logEntry);
	}


}
