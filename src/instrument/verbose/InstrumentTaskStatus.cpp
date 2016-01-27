#include <cassert>

#include "InstrumentTaskStatus.hpp"
#include "InstrumentVerbose.hpp"
#include "executors/threads/WorkerThread.hpp"


using namespace Instrument::Verbose;


namespace Instrument {
	void taskIsPending(task_id_t taskId) {
		if (!_verboseTaskStatus) {
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
		logEntry->_contents << " <-> TaskStatusChange " << taskId << " to:pending";
		
		addLogEntry(logEntry);
	}
	
	
	void taskIsReady(task_id_t taskId) {
		if (!_verboseTaskStatus) {
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
		logEntry->_contents << " <-> TaskStatusChange " << taskId << " to:ready";
		
		addLogEntry(logEntry);
	}
	
	
	void taskIsExecuting(task_id_t taskId) {
		if (!_verboseTaskStatus) {
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
		logEntry->_contents << " <-> TaskStatusChange " << taskId << " to:executing";
		
		addLogEntry(logEntry);
	}
	
	
	void taskIsBlocked(task_id_t taskId, task_blocking_reason_t reason) {
		if (!_verboseTaskStatus) {
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
		logEntry->_contents << " <-> TaskStatusChange " << taskId << " to:blocked";
		logEntry->_contents << " reason:";
		switch (reason) {
			case in_taskwait_blocking_reason:
				logEntry->_contents << "taskwait";
				break;
			case in_mutex_blocking_reason:
				logEntry->_contents << "mutex";
				break;
			default:
				logEntry->_contents << "unknown";
				break;
		}
		
		addLogEntry(logEntry);
	}
	
	
	void taskIsZombie(task_id_t taskId) {
		if (!_verboseTaskStatus) {
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
		logEntry->_contents << " <-> TaskStatusChange " << taskId << " to:zombie";
		
		addLogEntry(logEntry);
	}
	
	
}
