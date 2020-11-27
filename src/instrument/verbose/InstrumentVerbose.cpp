/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/


#include "InstrumentVerbose.hpp"


namespace Instrument {
	namespace Verbose {
		bool _verboseAddTask = false;
		bool _verboseBlocking = false;
		bool _verboseComputePlaceManagement = false;
		bool _verboseDependenciesAutomataMessages = false;
		bool _verboseDependenciesByAccess = false;
		bool _verboseDependenciesByAccessLinks = false;
		bool _verboseDependenciesByGroup = false;
		bool _verboseLeaderThread = false;
		bool _verboseReductions = false;
		bool _verboseTaskExecution = false;
		bool _verboseTaskStatus = false;
		bool _verboseTaskWait = false;
		bool _verboseThreadManagement = false;
		bool _verboseUserMutex = false;
		bool _verboseLoggingMessages = false;

		ConfigVariable<bool> _useTimestamps("instrument.verbose.timestamps");
		ConfigVariable<bool> _dumpOnlyOnExit("instrument.verbose.dump_only_on_exit");

		std::ofstream *_output = nullptr;

		ConcurrentUnorderedListSlotManager _concurrentUnorderedListSlotManager;
		ConcurrentUnorderedList<LogEntry *> _entries(_concurrentUnorderedListSlotManager);
		ConcurrentUnorderedList<LogEntry *> _freeEntries(_concurrentUnorderedListSlotManager);
		ConcurrentUnorderedListSlotManager::Slot _concurrentUnorderedListExternSlot;
	}
}
