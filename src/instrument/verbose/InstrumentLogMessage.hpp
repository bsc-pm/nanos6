#ifndef INSTRUMENT_VERBOSE_LOG_MESSAGE_HPP
#define INSTRUMENT_VERBOSE_LOG_MESSAGE_HPP


#include <cassert>

#include "../api/InstrumentLogMessage.hpp"
#include "executors/threads/WorkerThread.hpp"

#include "InstrumentVerbose.hpp"


using namespace Instrument::Verbose;


namespace Instrument {
	namespace Verbose {
		template<typename T>
		inline void fillLogEntry(LogEntry *logEntry, T contents)
		{
			logEntry->_contents << contents;
		}
		
		template<typename T, typename... TS>
		inline void fillLogEntry(LogEntry *logEntry, T content1, TS... contents)
		{
			logEntry->_contents << content1;
			fillLogEntry(logEntry, contents...);
		}
	}
	
	
	template<typename... TS>
	inline void logMessage(task_id_t triggererTaskId, TS... contents)
	{
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
		
		if (currentWorker != nullptr) {
			logEntry->_contents << "Thread:" << currentWorker << " CPU:" << currentWorker->getCpuId();
		} else {
			logEntry->_contents << "Thread:leader/unknown";
		}
		
		if (triggererTaskId != task_id_t()) {
			logEntry->_contents << " Task:" << triggererTaskId;
		}
		
		logEntry->_contents << " ";
		
		fillLogEntry(logEntry, contents...);
		
		addLogEntry(logEntry);
	}
}


#endif // INSTRUMENT_VERBOSE_LOG_MESSAGE_HPP
