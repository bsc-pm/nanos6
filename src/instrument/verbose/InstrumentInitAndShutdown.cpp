#ifndef INSTRUMENT_INIT_AND_SHUTDOWN_HPP
#define INSTRUMENT_INIT_AND_SHUTDOWN_HPP

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif

#include <time.h>


#include <algorithm>
#include <string>

#include <InstrumentLeaderThread.hpp>

#include "InstrumentVerbose.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/TokenizedEnvironmentVariable.hpp"


#define VERBOSE_INITIAL_LOG_ENTRIES 5000


using namespace Instrument::Verbose;


namespace Instrument {
	void initialize()
	{
		TokenizedEnvironmentVariable<std::string> verboseAreas("NANOS_VERBOSE", ',', "all");
		for (auto area : verboseAreas) {
			std::transform(area.begin(), area.end(), area.begin(), ::tolower);
			if (area == "all") {
				_verboseAddTask = true;
				_verboseDependenciesByAccess = true;
				_verboseDependenciesByAccessLinks = true;
				_verboseDependenciesByGroup = true;
				_verboseLeaderThread = true;
				_verboseTaskExecution = true;
				_verboseTaskStatus = true;
				_verboseTaskWait = true;
				_verboseThreadManagement = true;
				_verboseUserMutex = true;
				_verboseLoggingMessages = true;
			} else if (area == "addtask") {
				_verboseAddTask = true;
			} else if (area == "dependenciesbyaccess") {
				_verboseDependenciesByAccess = true;
			} else if (area == "dependenciesbyaccesslinks") {
				_verboseDependenciesByAccessLinks = true;
			} else if (area == "dependenciesbygroup") {
				_verboseDependenciesByGroup = true;
			} else if (area == "leaderthread") {
				_verboseLeaderThread = true;
			} else if (area == "taskexecution") {
				_verboseTaskExecution = true;
			} else if (area == "taskstatus") {
				_verboseTaskStatus = true;
			} else if (area == "taskwait") {
				_verboseTaskWait = true;
			} else if (area == "threadmanagement") {
				_verboseThreadManagement = true;
			} else if (area == "usermutex") {
				_verboseUserMutex = true;
			} else if (area == "logmessages") {
				_verboseLoggingMessages = true;
			
			} else if (area == "!addtask") {
				_verboseAddTask = false;
			} else if (area == "!dependenciesbyaccess") {
				_verboseDependenciesByAccess = false;
			} else if (area == "!dependenciesbyaccesslinks") {
				_verboseDependenciesByAccessLinks = false;
			} else if (area == "!dependenciesbygroup") {
				_verboseDependenciesByGroup = false;
			} else if (area == "!leaderthread") {
				_verboseLeaderThread = false;
			} else if (area == "!taskexecution") {
				_verboseTaskExecution = false;
			} else if (area == "!taskstatus") {
				_verboseTaskStatus = false;
			} else if (area == "!taskwait") {
				_verboseTaskWait = false;
			} else if (area == "!threadmanagement") {
				_verboseThreadManagement = false;
			} else if (area == "!usermutex") {
				_verboseUserMutex = false;
			} else if (area == "!logmessages") {
				_verboseLoggingMessages = false;
			
			} else {
				std::cerr << "Warning: ignoring unknown '" << area << "' verbose instrumentation" << std::endl;
			}
		}
		
		EnvironmentVariable<std::string> outputFilename("NANOS_VERBOSE_FILE", "/dev/stderr");
		_output = new std::ofstream(outputFilename.getValue().c_str());
		
		// Prepopulate the list of free log entries
		LogEntry *lastEntry = new LogEntry();
		lastEntry->_next = nullptr;
		for (int i = 1; i < VERBOSE_INITIAL_LOG_ENTRIES; i++) {
			LogEntry *newEntry = new LogEntry();
			newEntry->_next = lastEntry;
			lastEntry = newEntry;
		}
		_freeEntries.store(lastEntry);
	}
	
	
	void shutdown()
	{
		// Flush out any pending log entries
		Instrument::leaderThreadSpin();
		
		_output->close();
		
		// TODO: Free up all the memory
	}
	
}


#endif // INSTRUMENT_INIT_AND_SHUTDOWN_HPP
