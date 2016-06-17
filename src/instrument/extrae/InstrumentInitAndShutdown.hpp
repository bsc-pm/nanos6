#ifndef INSTRUMENT_EXTRAE_INIT_AND_SHUTDOWN_HPP
#define INSTRUMENT_EXTRAE_INIT_AND_SHUTDOWN_HPP

#include <cassert>

#include "api/nanos6_debug_interface.h"
#include "executors/threads/ThreadManager.hpp"
#include "../InstrumentInitAndShutdown.hpp"
#include "InstrumentExtrae.hpp"


namespace Instrument {
	static unsigned int extrae_nanos_get_num_cpus()
	{
		// CPU "0" is for the leader thread
		return nanos_get_num_cpus() + 1;
	}
	
	void initialize()
	{
		// Common thread information callbacks
		Extrae_set_threadid_function ( nanos_get_current_virtual_cpu );
		Extrae_set_numthreads_function ( extrae_nanos_get_num_cpus );
		
		// Initialize extrae library
		Extrae_init();
		
		Extrae_register_codelocation_type( _functionName, _codeLocation, "User Function Name", "User Function Location" );
		Extrae_register_stacked_type( (extrae_type_t) _runtimeState );
		Extrae_register_stacked_type( (extrae_type_t) _functionName );
		Extrae_register_stacked_type( (extrae_type_t) _codeLocation );
	}
	
	
	void shutdown()
	{
		unsigned int nval = NANOS_EVENT_STATE_TYPES;
		extrae_value_t values[nval];
		unsigned int i;
		
		for ( i = 0; i < nval; i++ ) values[i] = i;
		
		Extrae_define_event_type( (extrae_type_t *) &_runtimeState, (char *) "Runtime state", &nval, values, _eventStateValueStr );
		
		for (nanos_task_info *taskInfo : _userFunctionMap) {
			assert(taskInfo != nullptr);
			
			Extrae_register_function_address (
				(void *) (taskInfo->run),
				(taskInfo->task_label != nullptr ? taskInfo->task_label : taskInfo->declaration_source),
				taskInfo->declaration_source, 0
			);
		}
		
		// Finalize extrae library
		Extrae_fini();
	}
}


#endif // INSTRUMENT_EXTRAE_INIT_AND_SHUTDOWN_HPP
