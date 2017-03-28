#ifndef INSTRUMENT_EXTRAE_INIT_AND_SHUTDOWN_HPP
#define INSTRUMENT_EXTRAE_INIT_AND_SHUTDOWN_HPP

#include <cassert>

#include <nanos6/debug.h>

#include "../api/InstrumentInitAndShutdown.hpp"
#include "../generic_ids/GenericIds.hpp"
#include "InstrumentExtrae.hpp"


namespace Instrument {
	static unsigned int extrae_nanos_get_num_cpus()
	{
		return nanos_get_num_cpus();
	}
	
	static unsigned int extrae_nanos_get_thread_id()
	{
		if (_currentThreadId != nullptr) {
			return *_currentThreadId;
		} else {
			return 0;
		}
	}
	
	void initialize()
	{
		// Common thread information callbacks
		if (_traceAsThreads) {
			Extrae_set_threadid_function ( extrae_nanos_get_thread_id );
			Extrae_set_numthreads_function ( extrae_nanos_get_num_threads );
		} else {
			Extrae_set_threadid_function ( nanos_get_current_virtual_cpu );
			Extrae_set_numthreads_function ( extrae_nanos_get_num_cpus );
		}
		
		// Initialize extrae library
		Extrae_init();
		
		unsigned int zero = 0;
		
		Extrae_register_codelocation_type( _functionName, _codeLocation, "User Function Name", "User Function Location" );
		Extrae_define_event_type((extrae_type_t *) &_taskInstanceId, "Task instance", &zero, nullptr, nullptr);
		Extrae_define_event_type((extrae_type_t *) &_nestingLevel, "Task nesting level", &zero, nullptr, nullptr);
		
		Extrae_register_stacked_type( (extrae_type_t) _runtimeState );
		Extrae_register_stacked_type( (extrae_type_t) _functionName );
		Extrae_register_stacked_type( (extrae_type_t) _codeLocation );
		Extrae_register_stacked_type( (extrae_type_t) _taskInstanceId );
		Extrae_register_stacked_type( (extrae_type_t) _nestingLevel );
	}
	
	
	void shutdown()
	{
		unsigned int nval = NANOS_EVENT_STATE_TYPES;
		extrae_value_t values[nval];
		unsigned int i;
		
		for ( i = 0; i < nval; i++ ) values[i] = i;
		
		Extrae_define_event_type( (extrae_type_t *) &_runtimeState, (char *) "Runtime state", &nval, values, _eventStateValueStr );
		
		std::set<nanos_task_info *, ExtraeTaskInfoCompare> orderedTaskInfoMap(_userFunctionMap.begin(), _userFunctionMap.end());
		for (nanos_task_info *taskInfo : orderedTaskInfoMap) {
			assert(taskInfo != nullptr);
			
			if (taskInfo->run != nullptr) {
				Extrae_register_function_address (
					(void *) (taskInfo->run),
					(taskInfo->task_label != nullptr ? taskInfo->task_label : taskInfo->declaration_source),
					taskInfo->declaration_source, 0
				);
			}
		}
		
		// Finalize extrae library
		Extrae_fini();
	}
}


#endif // INSTRUMENT_EXTRAE_INIT_AND_SHUTDOWN_HPP
