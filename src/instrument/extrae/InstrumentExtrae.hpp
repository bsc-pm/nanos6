#ifndef INSTRUMENT_EXTRAE_HPP
#define INSTRUMENT_EXTRAE_HPP

#include "extrae_user_events.h"
#include "extrae_types.h"

#include "api/nanos6_rt_interface.h"
#include "lowlevel/SpinLock.hpp"

#include <set>
#include <map>


namespace Instrument {
	typedef std::set<nanos_task_info *>              user_fct_map_t;
	
	extern const extrae_type_t                       _runtimeState;      //!< Runtime state (extrae event type)
	extern const extrae_type_t                       _functionName;      //!< Task function name
	extern const extrae_type_t                       _codeLocation;      //!< Task function location
	
	typedef enum { NANOS_NO_STATE, NANOS_NOT_RUNNING, NANOS_STARTUP, NANOS_SHUTDOWN, NANOS_ERROR, NANOS_IDLE,
						NANOS_RUNTIME, NANOS_RUNNING, NANOS_SYNCHRONIZATION, NANOS_SCHEDULING, NANOS_CREATION,
						NANOS_EVENT_STATE_TYPES
	} nanos_event_state_t;
	
	extern char                                     *_eventStateValueStr[];
	
	extern SpinLock _userFunctionMapLock;
	extern user_fct_map_t                            _userFunctionMap;
	
}

#endif
