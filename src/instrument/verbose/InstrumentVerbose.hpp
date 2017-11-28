/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

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

#include <InstrumentInstrumentationContext.hpp>

#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

#include <support/ConcurrentUnorderedList.hpp>


namespace Instrument {
	namespace Verbose {
		typedef struct timespec timestamp_t;
		
		extern bool _verboseAddTask;
		extern bool _verboseBlocking;
		extern bool _verboseComputePlaceManagement;
		extern bool _verboseDependenciesByAccess;
		extern bool _verboseDependenciesByAccessLinks;
		extern bool _verboseDependenciesByGroup;
		extern bool _verboseLeaderThread;
		extern bool _verboseReductions;
		extern bool _verboseTaskExecution;
		extern bool _verboseTaskStatus;
		extern bool _verboseTaskWait;
		extern bool _verboseThreadManagement;
		extern bool _verboseUserMutex;
		extern bool _verboseLoggingMessages;
		
		extern EnvironmentVariable<bool> _useTimestamps;
		extern EnvironmentVariable<bool> _dumpOnlyOnExit;
		
		extern std::ofstream *_output;
		
		
		struct LogEntry {
			timestamp_t _timestamp;
			ConcurrentUnorderedListSlotManager::Slot _queueSlot;
			std::ostringstream _contents;
			
			LogEntry(timestamp_t timestamp, ConcurrentUnorderedListSlotManager::Slot queueSlot, std::ostringstream const &contents)
				: _timestamp(timestamp), _queueSlot(queueSlot), _contents()
			{
				_contents << contents.str();
			}
			
			LogEntry(ConcurrentUnorderedListSlotManager::Slot queueSlot)
				: _queueSlot(queueSlot)
			{
			}
			
			void appendLocation(InstrumentationContext const &context)
			{
				if (context._externalThreadName != nullptr) {
					_contents << "ExternalThread:" << *context._externalThreadName;
				} else {
					assert(context._threadId != thread_id_t());
					
					_contents << "Thread:" << context._threadId << " ComputePlace:";
					if (context._computePlaceId != compute_place_id_t()) {
						_contents << context._computePlaceId;
					} else {
						_contents << "unknown";
					}
				}
			}
		};
		
		extern ConcurrentUnorderedListSlotManager _concurrentUnorderedListSlotManager;
		extern ConcurrentUnorderedList<LogEntry *> _entries;
		extern ConcurrentUnorderedList<LogEntry *> _freeEntries;
		extern ConcurrentUnorderedListSlotManager::Slot _concurrentUnorderedListExternSlot;
		
		
		static inline void stampTime(LogEntry *logEntry)
		{
			assert(logEntry != nullptr);
			if (_useTimestamps) {
				int rc = clock_gettime(CLOCK_MONOTONIC, &logEntry->_timestamp);
				FatalErrorHandler::handle(rc, "Retrieving the monotonic time");
			}
		}
		
		
		inline LogEntry *getLogEntry(InstrumentationContext const &context)
		{
			ConcurrentUnorderedListSlotManager::Slot queueSlot = _concurrentUnorderedListExternSlot;
			if (context._externalThreadName == nullptr) {
				queueSlot = context._computePlaceId.getConcurrentUnorderedListSlot();
			}
			
			LogEntry *currentEntry = nullptr;
			if (_freeEntries.pop(currentEntry, queueSlot)) {
				assert(currentEntry != nullptr);
				assert(currentEntry->_queueSlot == queueSlot);
				currentEntry->_contents.clear();
			} else {
				currentEntry = new LogEntry(queueSlot);
			}
			
			stampTime(currentEntry);
			return currentEntry;
		}
		
		
		inline void addLogEntry(LogEntry *logEntry)
		{
			assert(logEntry != nullptr);
			
			_entries.push(logEntry, logEntry->_queueSlot);
		}
		
	}
}


#endif // INSTRUMENT_VERBOSE_HPP
