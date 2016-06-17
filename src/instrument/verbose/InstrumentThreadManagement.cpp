#include <cassert>

#include "InstrumentThreadManagement.hpp"
#include "InstrumentVerbose.hpp"
#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"

#include "../generic_ids/GenericIds.hpp"


using namespace Instrument::Verbose;


namespace Instrument {
	thread_id_t createdThread() {
		thread_id_t::inner_type_t threadId = GenericIds::getNewThreadId();
		
		if (!_verboseThreadManagement) {
			return thread_id_t(threadId);
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
		
		if (currentWorker != nullptr) {
			logEntry->_contents << "Thread:" << currentWorker << " CPU:" << currentWorker->getCpuId();
		} else {
			logEntry->_contents << "Thread:LeaderThread CPU:ANY";
		}
		
		logEntry->_contents << " <-> CreateThread " << threadId;
		
		addLogEntry(logEntry);
		
		return thread_id_t(threadId);
	}
	
	
	void threadWillSuspend(thread_id_t threadId, cpu_id_t cpuId) {
		if (!_verboseThreadManagement) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->_contents << "Thread:" << (thread_id_t::inner_type_t) threadId << " CPU:" << (cpu_id_t::inner_type_t) cpuId << " --> SuspendThread ";
		
		addLogEntry(logEntry);
	}
	
	
	void threadHasResumed(thread_id_t threadId, cpu_id_t cpuId) {
		if (!_verboseThreadManagement) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		logEntry->_contents << "Thread:" << (thread_id_t::inner_type_t) threadId << " CPU:" << (cpu_id_t::inner_type_t) cpuId << " <-- SuspendThread ";
		
		addLogEntry(logEntry);
	}
	
	
}
