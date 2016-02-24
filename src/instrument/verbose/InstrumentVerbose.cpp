#include "InstrumentVerbose.hpp"


namespace Instrument {
	namespace Verbose {
		bool _verboseAddTask = false;
		bool _verboseDependenciesByAccess = false;
		bool _verboseDependenciesByAccessLinks = false;
		bool _verboseDependenciesByGroup = false;
		bool _verboseLeaderThread = true;
		bool _verboseTaskExecution = false;
		bool _verboseTaskStatus = false;
		bool _verboseTaskWait = false;
		bool _verboseThreadManagement = false;
		bool _verboseUserMutex = false;
		
		EnvironmentVariable<bool> _useTimestamps("NANOS_VERBOSE_TIMESTAMPS", true);
		
		std::ofstream *_output = nullptr;
		
		std::atomic<LogEntry *> _lastEntry(nullptr);
		std::atomic<LogEntry *> _freeEntries(nullptr);
	}
}
