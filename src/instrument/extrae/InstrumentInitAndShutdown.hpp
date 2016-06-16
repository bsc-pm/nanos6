#ifndef INSTRUMENT_EXTRAE_INIT_AND_SHUTDOWN_HPP
#define INSTRUMENT_EXTRAE_INIT_AND_SHUTDOWN_HPP


#include "executors/threads/ThreadManager.hpp"
#include "../InstrumentInitAndShutdown.hpp"
#include "InstrumentExtrae.hpp"


extern "C" {
	unsigned int nanos_get_thread_num ( void ) 
	{
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		
		unsigned int cpuId = 0;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getHardwarePlace();
			cpuId = cpu->_virtualCPUId;
		}
		
		return (unsigned int) cpuId;
	}
	
	
	unsigned int nanos_get_max_threads ( void )
	{
		unsigned int retval = 0; 
		if (ThreadManager::hasFinishedInitialization()) {
			retval = ThreadManager::getTotalCPUs();
		} else {
			cpu_set_t const &cpuMask = ThreadManager::getProcessCPUMaskReference();
			for (size_t systemCPUId = 0; systemCPUId < CPU_SETSIZE; systemCPUId++) {
				if (CPU_ISSET(systemCPUId, &cpuMask)) {
					retval++;
				}
			}
		}
		
		return retval;
	}
};


namespace Instrument {
	void initialize()
	{
		// Assign thread identifier 0 to the leader thread
		_threadToId[0] = 0;
		
		// Common thread information callbacks
		Extrae_set_threadid_function ( nanos_get_thread_num );
		Extrae_set_numthreads_function ( nanos_get_max_threads );
		
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
		unsigned int nzero = 0;
		extrae_value_t values[nval];
		unsigned int i;
		
		for ( i = 0; i < nval; i++ ) values[i] = i;
		
		Extrae_define_event_type( (extrae_type_t *) &_runtimeState, (char *) "Runtime state", &nval, values, _eventStateValueStr );
		
		user_fct_map_t::iterator it;
		
		for ( it = _userFunctionMap.begin(); it != _userFunctionMap.end(); it++ ) {
			//FIXME:XXX substitute address
			Extrae_register_function_address ( (void *) (it->first), (char *) (it->second), (char *) (it->second), (unsigned) 13); 
		}
		
		// Finalize extrae library
		Extrae_fini();
	}
}


#endif // INSTRUMENT_EXTRAE_INIT_AND_SHUTDOWN_HPP
