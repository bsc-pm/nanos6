#include <cassert>

#include "InstrumentTaskExecution.hpp"
#include "InstrumentVerbose.hpp"
#include "executors/threads/CPU.hpp"


using namespace Instrument::Verbose;


namespace Instrument {
	void startTask(task_id_t taskId, cpu_id_t cpuId, thread_id_t currentThreadId) {
		if (!_verboseTaskExecution) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->_contents << "Thread:" << currentThreadId << " CPU:" << cpuId << " --> Task " << taskId;
		
		addLogEntry(logEntry);
	}
	
	
	void returnToTask(task_id_t taskId, cpu_id_t cpuId, thread_id_t currentThreadId) {
		if (!_verboseTaskExecution) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->_contents << "Thread:" << currentThreadId << " CPU:" << cpuId << " ->> Task " << taskId;
		
		addLogEntry(logEntry);
	}
	
	
	void endTask(task_id_t taskId, cpu_id_t cpuId, thread_id_t currentThreadId) {
		if (!_verboseTaskExecution) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->_contents << "Thread:" << currentThreadId << " CPU:" << cpuId << " <-- Task " << taskId;
		
		addLogEntry(logEntry);
	}
	
	
	void destroyTask(task_id_t taskId, cpu_id_t cpuId, thread_id_t currentThreadId) {
		if (!_verboseTaskExecution) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->_contents << "Thread:" << currentThreadId << " CPU:" << cpuId << " <-> DestroyTask " << taskId;
		
		addLogEntry(logEntry);
	}
	
}
