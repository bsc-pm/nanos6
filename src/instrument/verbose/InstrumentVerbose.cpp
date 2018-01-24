/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupport.hpp>
#include <instrument/support/InstrumentThreadLocalDataSupportImplementation.hpp>

#include "InstrumentVerbose.hpp"


namespace Instrument {
	namespace Verbose {
		bool _verboseAddTask = false;
		bool _verboseComputePlaceManagement = false;
		bool _verboseDependenciesByAccess = false;
		bool _verboseDependenciesByAccessLinks = false;
		bool _verboseDependenciesByGroup = false;
		bool _verboseLeaderThread = true;
		bool _verboseTaskExecution = false;
		bool _verboseTaskStatus = false;
		bool _verboseTaskWait = false;
		bool _verboseThreadManagement = false;
		bool _verboseUserMutex = false;
		bool _verboseLoggingMessages = false;
		
		EnvironmentVariable<bool> _useTimestamps("NANOS6_VERBOSE_TIMESTAMPS", true);
		EnvironmentVariable<bool> _dumpOnlyOnExit("NANOS6_VERBOSE_DUMP_ONLY_ON_EXIT", false);
		
		std::ofstream *_output = nullptr;
		
		ConcurrentUnorderedListSlotManager _concurrentUnorderedListSlotManager;
		ConcurrentUnorderedList<LogEntry *> _entries(_concurrentUnorderedListSlotManager);
		ConcurrentUnorderedList<LogEntry *> _freeEntries(_concurrentUnorderedListSlotManager);
		ConcurrentUnorderedListSlotManager::Slot _concurrentUnorderedListExternSlot;
	}
}
