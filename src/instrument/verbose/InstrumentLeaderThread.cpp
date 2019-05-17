/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <mutex>
#include <vector>

#include "InstrumentLeaderThread.hpp"
#include "InstrumentVerbose.hpp"
#include "lowlevel/SpinLock.hpp"
#include "system/LeaderThread.hpp"

#include <instrument/support/InstrumentThreadLocalDataSupport.hpp>

#ifdef __ANDROID__
#include <android/log.h>
#endif

using namespace Instrument::Verbose;


namespace Instrument {
	void leaderThreadSpin() {
		static SpinLock lock;
		
		// This is needed since this method can be called by a regular thread on abort
		std::lock_guard<SpinLock> guard(lock);
		
		// Logging part
		if (_verboseLeaderThread) {
			ExternalThreadLocalData &threadLocal = getExternalThreadLocalData();
			
			LogEntry *logEntry = getLogEntry(threadLocal._context);
			assert(logEntry != nullptr);
			
			logEntry->appendLocation(threadLocal._context);
			logEntry->_contents << " --- LeaderThreadSpin";
			
			addLogEntry(logEntry);
		}
		
		if (_dumpOnlyOnExit && !LeaderThread::isExiting()) {
			return;
		}
		
		// After this we flush the current log
		
		// Move the log to a vector
		static std::vector<LogEntry *> entries;
		_entries.consume_all(
			[&](LogEntry *entry, __attribute__((unused)) ConcurrentUnorderedListSlotManager::Slot slot)
			{
				entries.push_back(entry);
			}
		);
		
		// If using timestamps, sort the vector
		if (_useTimestamps) {
			struct {
				bool operator()(LogEntry *a, LogEntry *b)
				{
					assert(a != nullptr);
					assert(b != nullptr);
					if (a->_timestamp.tv_sec < b->_timestamp.tv_sec) {
						return true;
					} else if (a->_timestamp.tv_sec == b->_timestamp.tv_sec) {
						return (a->_timestamp.tv_nsec < b->_timestamp.tv_nsec);
					} else {
						return false;
					}
				}
			} comparator;
			std::sort(entries.begin(), entries.end(), comparator);
		}
		
		// Dump the log and leave the entries ready to be reused
#ifndef __ANDROID__
		assert(_output != nullptr);
#endif
		if (_useTimestamps) {
			// Dump the log from the vector, which is already sorted by timestamp
			for (LogEntry *logEntry : entries) {
				assert(logEntry != nullptr);
				
#ifdef __ANDROID__
				if (_output == nullptr) {
					__android_log_print(ANDROID_LOG_DEBUG, "Nanos6", "%lu.%09lu %s\n",
										logEntry->_timestamp.tv_sec, logEntry->_timestamp.tv_nsec,
										logEntry->_contents.str().c_str());
				} else {
#endif
				(*_output) << logEntry->_timestamp.tv_sec << "." << std::setw(9) << std::setfill('0') << logEntry->_timestamp.tv_nsec << std::setw(0) << std::setfill(' ');
				(*_output) << " " << logEntry->_contents.str() << std::endl;
#ifdef __ANDROID__
				}
#endif
				logEntry->_contents.str("");
			}
		} else {
			// Dump the (unsorted) vector in reverse order, which is the actual order of the log creation
			for (auto it = entries.rbegin(); it != entries.rend(); it++) {
				LogEntry *logEntry = *it;
#ifdef __ANDROID__
				if (_output == nullptr) {
					__android_log_print(ANDROID_LOG_DEBUG, "Nanos6", "%s\n", logEntry->_contents.str().c_str());
				} else {
#endif
				(*_output) << logEntry->_contents.str() << std::endl;
#ifdef __ANDROID__
				}
#endif
				logEntry->_contents.str("");
			}
		}
		
		// Recycle the entries
		for (LogEntry *logEntry : entries) {
			_freeEntries.push(logEntry, logEntry->_queueSlot);
		}
		
		// Clear the vector, since we reuse it
		entries.clear();
	}
	
	
}
