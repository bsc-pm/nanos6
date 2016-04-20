#ifndef INSTRUMENT_EXTRAE_HPP
#define INSTRUMENT_EXTRAE_HPP

#include "extrae_user_events.h"
#include "extrae_types.h"

#include "lowlevel/SpinLock.hpp"
#include <map>

class WorkerThread;

namespace Instrument {

typedef long                                     thread_id_t;
typedef std::map<WorkerThread *, thread_id_t>    thread_map_t;
typedef std::map<void *, const char *>           user_fct_map_t;

extern const extrae_type_t                       _runtimeState;      //!< Runtime state (extrae event type)
extern const extrae_type_t                       _functionName;      //!< Task function name
extern const extrae_type_t                       _codeLocation;      //!< Task function location
extern thread_map_t                              _threadToId;        //!< maps thread pointers to thread identifiers
		
typedef enum { NANOS_NO_STATE, NANOS_NOT_RUNNING, NANOS_STARTUP, NANOS_SHUTDOWN, NANOS_ERROR, NANOS_IDLE,
               NANOS_RUNTIME, NANOS_RUNNING, NANOS_SYNCHRONIZATION, NANOS_SCHEDULING, NANOS_CREATION,
               NANOS_EVENT_STATE_TYPES
} nanos_event_state_t;

extern char                                     *_eventStateValueStr[];
extern user_fct_map_t                            _userFunctionMap;

}

#endif
