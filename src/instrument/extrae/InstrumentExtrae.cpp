#include "InstrumentExtrae.hpp"

#include "../generic_ids/GenericIds.hpp"

#include <nanos6/debug.h>


namespace Instrument {
	const EnvironmentVariable<bool> _traceAsThreads("NANOS_EXTRAE_AS_THREADS", 0);
	
	const extrae_type_t       _taskInstanceId = 9200002;
	const extrae_type_t       _runtimeState = 9000000;     //!< Runtime state (extrae event type)
	const extrae_type_t       _functionName = 9200011;     //!< Task function name
	const extrae_type_t       _codeLocation = 9200021;     //!< Task code location
	const extrae_type_t       _nestingLevel = 9500001;     //!< Nesting level
	
	SpinLock                  _extraeLock;
	
	char const               *_eventStateValueStr[NANOS_EVENT_STATE_TYPES] = {
		"NOT CREATED", "NOT RUNNING", "STARTUP", "SHUTDOWN", "ERROR", "IDLE",
		"RUNTIME", "RUNNING", "SYNCHRONIZATION", "SCHEDULING", "CREATION" };
	
	SpinLock _userFunctionMapLock;
	user_fct_map_t            _userFunctionMap;
	
	std::atomic<size_t> _nextTaskId(1);
	
	RWSpinLock _extraeThreadCountLock;
	
	unsigned int extrae_nanos_get_num_threads()
	{
		if (_traceAsThreads) {
			return GenericIds::getTotalThreads() + 1 /* Starts counting at 1 */;
		} else {
			return nanos_get_num_cpus();
		}
	}

}

