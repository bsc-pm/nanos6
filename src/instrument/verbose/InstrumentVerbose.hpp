#ifndef INSTRUMENT_VERBOSE_HPP
#define INSTRUMENT_VERBOSE_HPP

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif

#include <time.h>


#include <atomic>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


namespace Instrument {
	namespace Verbose {
		typedef struct timespec timestamp_t;
		
		extern bool _verboseAddTask;
		extern bool _verboseDependenciesByAccess;
		extern bool _verboseDependenciesByAccessSequence;
		extern bool _verboseDependenciesByGroup;
		extern bool _verboseLeaderThread;
		extern bool _verboseTaskExecution;
		extern bool _verboseTaskStatus;
		extern bool _verboseTaskWait;
		extern bool _verboseThreadManagement;
		extern bool _verboseUserMutex;
		
		extern EnvironmentVariable<bool> _useTimestamps;
		
		extern std::ofstream *_output;
		
		
		struct LogEntry {
			timestamp_t _timestamp;
			std::ostringstream _contents;
			LogEntry *_next;
			
			LogEntry(timestamp_t timestamp, std::ostringstream &&contents)
				: _timestamp(timestamp), _contents(std::move(contents)), _next(nullptr)
			{
			}
			
			LogEntry()
				: _next(nullptr)
			{
			}
		};
		
		extern std::atomic<LogEntry *> _lastEntry;
		extern std::atomic<LogEntry *> _freeEntries;
		
		
		static inline void stampTime(LogEntry *logEntry)
		{
			assert(logEntry != nullptr);
			if (_useTimestamps) {
				int rc = clock_gettime(CLOCK_MONOTONIC, &logEntry->_timestamp);
				FatalErrorHandler::handle(rc, "Retrieving the monotonic time");
			}
		}
		
		
		inline LogEntry *getLogEntry()
		{
			LogEntry *currentEntry = _freeEntries;
			while (currentEntry != nullptr) {
				LogEntry *nextEntry = currentEntry->_next;
				if (_freeEntries.compare_exchange_weak(currentEntry, nextEntry)) {
					assert(currentEntry != nullptr);
					currentEntry->_contents.clear();
					stampTime(currentEntry);
					return currentEntry;
				}
			}
			
			currentEntry = new LogEntry();
			stampTime(currentEntry);
			return currentEntry;
		}
		
		
		inline void addLogEntry(LogEntry *logEntry)
		{
			assert(logEntry != nullptr);
			
			LogEntry *lastEntry = _lastEntry;
			do {
				logEntry->_next = lastEntry;
			} while (!_lastEntry.compare_exchange_weak(lastEntry, logEntry, std::memory_order_release));
		}
		
	}
}


#endif // INSTRUMENT_VERBOSE_HPP
