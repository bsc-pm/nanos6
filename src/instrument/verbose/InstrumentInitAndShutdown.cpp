/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

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
#include "support/config/ConfigVariable.hpp"
#include "system/RuntimeInfo.hpp"


#define VERBOSE_INITIAL_LOG_ENTRIES 5000


using namespace Instrument::Verbose;


namespace Instrument {
	void initialize()
	{
		RuntimeInfo::addEntry("instrumentation", "Instrumentation", "verbose");

		ConfigVariableSet<std::string> verboseAreas("instrument.verbose.areas");

		for (auto area : verboseAreas) {
			std::transform(area.begin(), area.end(), area.begin(), ::tolower);
			if (area == "all") {
				_verboseAddTask = true;
				_verboseBlocking = true;
				_verboseComputePlaceManagement = true;
				_verboseDependenciesAutomataMessages = true;
				_verboseDependenciesByAccess = true;
				_verboseDependenciesByAccessLinks = true;
				_verboseDependenciesByGroup = true;
				_verboseLeaderThread = true;
				_verboseReductions = true;
				_verboseTaskExecution = true;
				_verboseTaskStatus = true;
				_verboseTaskWait = true;
				_verboseThreadManagement = true;
				_verboseUserMutex = true;
				_verboseLoggingMessages = true;
			} else if (area == "addtask") {
				_verboseAddTask = true;
			} else if (area == "blocking") {
				_verboseBlocking = true;
			} else if (area == "computeplacemanagement") {
				_verboseComputePlaceManagement = true;
			} else if (area == "dependenciesautomatamessages") {
				_verboseDependenciesAutomataMessages = true;
			} else if (area == "dependenciesbyaccess") {
				_verboseDependenciesByAccess = true;
			} else if (area == "dependenciesbyaccesslinks") {
				_verboseDependenciesByAccessLinks = true;
			} else if (area == "dependenciesbygroup") {
				_verboseDependenciesByGroup = true;
			} else if (area == "leaderthread") {
				_verboseLeaderThread = true;
			} else if (area == "reductions") {
				_verboseReductions = true;
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
			} else if (area == "!blocking") {
				_verboseBlocking = false;
			} else if (area == "!computeplacemanagement") {
				_verboseComputePlaceManagement = false;
			} else if (area == "!dependenciesautomatamessages") {
				_verboseDependenciesAutomataMessages = false;
			} else if (area == "!dependenciesbyaccess") {
				_verboseDependenciesByAccess = false;
			} else if (area == "!dependenciesbyaccesslinks") {
				_verboseDependenciesByAccessLinks = false;
			} else if (area == "!dependenciesbygroup") {
				_verboseDependenciesByGroup = false;
			} else if (area == "!leaderthread") {
				_verboseLeaderThread = false;
			} else if (area == "!reductions") {
				_verboseReductions = false;
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

		ConfigVariable<std::string> outputFilename("instrument.verbose.output_file");
#ifdef __ANDROID__
		if (!outputFilename.isPresent()) {
			_output = nullptr;
		} else {
#endif
			_output = new std::ofstream(outputFilename.getValue().c_str());
#ifdef __ANDROID__
		}
#endif

		_concurrentUnorderedListExternSlot = _concurrentUnorderedListSlotManager.getSlot();
	}


	void shutdown()
	{
		// Flush out any pending log entries
		Instrument::leaderThreadSpin();

#ifdef __ANDROID__
		if (_output != nullptr) {
#endif
			_output->close();
#ifdef __ANDROID__
		}
#endif

		// TODO: Free up all the memory
	}

	void preinitFinished()
	{
	}

	int64_t getInstrumentStartTime()
	{
		return 0;
	}
} // namespace Instrument


#endif // INSTRUMENT_INIT_AND_SHUTDOWN_HPP
