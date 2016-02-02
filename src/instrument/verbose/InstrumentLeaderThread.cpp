#include <cassert>

#include "InstrumentLeaderThread.hpp"
#include "InstrumentVerbose.hpp"
#include "executors/threads/WorkerThread.hpp"

#include <algorithm>
#include <iomanip>
#include <vector>


using namespace Instrument::Verbose;


namespace Instrument {
	void leaderThreadSpin() {
		// Logging part
		if (_verboseLeaderThread) {
			LogEntry *logEntry = getLogEntry();
			assert(logEntry != nullptr);
			
			WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
			
			if (currentWorker != nullptr) {
			logEntry->_contents << "Thread:" << currentWorker << " CPU:" << currentWorker->getCpuId();
			} else {
				logEntry->_contents << "Thread:LeaderThread CPU:ANY";
			}
			logEntry->_contents << " --- LeaderThreadSpin";
			
			addLogEntry(logEntry);
		}
		
		// After this we flush the current log
		
		// Swap the current log
		LogEntry *currentEntry = _lastEntry.load();
		while (!_lastEntry.compare_exchange_weak(currentEntry, nullptr)) {
		}
		
		// Move the log to a vector
		static std::vector<LogEntry *> entries;
		LogEntry *firstEntry = currentEntry;
		LogEntry *lastEntry = currentEntry;
		while (currentEntry != nullptr) {
			entries.push_back(currentEntry);
			lastEntry = currentEntry;
			currentEntry = currentEntry->_next;
		}
		
		if (firstEntry == nullptr) {
			return;
		}
		assert(lastEntry != nullptr);
		
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
		assert(_output != nullptr);
		if (_useTimestamps) {
			// Dump the log from the vector, which is already sorted by timestamp
			for (LogEntry *logEntry : entries) {
				assert(logEntry != nullptr);
				
				(*_output) << logEntry->_timestamp.tv_sec << "." << std::setw(9) << std::setfill('0') << logEntry->_timestamp.tv_nsec << std::setw(0) << std::setfill(' ');
				(*_output) << " " << logEntry->_contents.str() << std::endl;
				logEntry->_contents.str("");
			}
		} else {
			// Dump the (unsorted) vector in reverse order, which is the actual order of the log creation
			for (auto it = entries.rbegin(); it != entries.rend(); it++) {
				LogEntry *logEntry = *it;
				(*_output) << logEntry->_contents.str() << std::endl;
				logEntry->_contents.str("");
			}
		}
		
		// Prepend the processed log entries to the list of free entries
		LogEntry *firstFreeEntry = _freeEntries;
		do {
			lastEntry->_next = firstFreeEntry;
		} while (!_freeEntries.compare_exchange_weak(firstFreeEntry, firstEntry));
		
		// Clean the vector, which is reused
		entries.clear();
	}
	
	
}
