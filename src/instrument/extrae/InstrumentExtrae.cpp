#include "InstrumentExtrae.hpp"

namespace Instrument {
	const extrae_type_t       _runtimeState = 9000000;     //!< Runtime state (extrae event type)
	const extrae_type_t       _functionName = 9200011;     //!< Task function name
	const extrae_type_t       _codeLocation = 9200021;     //!< Task code location
	
	SpinLock                  _extraeLock;
	
	char                     *_eventStateValueStr[NANOS_EVENT_STATE_TYPES] = {
		"NOT CREATED", "NOT RUNNING", "STARTUP", "SHUTDOWN", "ERROR", "IDLE",
		"RUNTIME", "RUNNING", "SYNCHRONIZATION", "SCHEDULING", "CREATION" };
	
	SpinLock _userFunctionMapLock;
	user_fct_map_t            _userFunctionMap;
}
