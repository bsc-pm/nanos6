#include <cassert>

#include "InstrumentTaskExecution.hpp"
#include "InstrumentVerbose.hpp"
#include "executors/threads/CPU.hpp"


using namespace Instrument::Verbose;


namespace Instrument {
	void startTask(task_id_t taskId, CPU *cpu, WorkerThread *currentThread) {
		if (!_verboseTaskExecution) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->_contents << "Thread:" << currentThread << " CPU:" << cpu->_virtualCPUId << " --> Task " << taskId;
		
		addLogEntry(logEntry);
	}
	
	
	void returnToTask(task_id_t taskId, CPU *cpu, WorkerThread *currentThread) {
		if (!_verboseTaskExecution) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->_contents << "Thread:" << currentThread << " CPU:" << cpu->_virtualCPUId << " ->> Task " << taskId;
		
		addLogEntry(logEntry);
	}
	
	
	void endTask(task_id_t taskId, CPU *cpu, WorkerThread *currentThread) {
		if (!_verboseTaskExecution) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->_contents << "Thread:" << currentThread << " CPU:" << cpu->_virtualCPUId << " <-- Task " << taskId;
		
		addLogEntry(logEntry);
	}
	
	
	void destroyTask(task_id_t taskId, CPU *cpu, WorkerThread *currentThread) {
		if (!_verboseTaskExecution) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->_contents << "Thread:" << currentThread << " CPU:" << cpu->_virtualCPUId << " <-> DestroyTask " << taskId;
		
		addLogEntry(logEntry);
	}


}
