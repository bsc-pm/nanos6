/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_INIT_AND_SHUTDOWN_HPP
#define INSTRUMENT_EXTRAE_INIT_AND_SHUTDOWN_HPP

#include <cassert>
#include <cstdlib>

#include <dlfcn.h>

#include <nanos6/debug.h>

#include "../api/InstrumentInitAndShutdown.hpp"
#include "../generic_ids/GenericIds.hpp"
#include "system/RuntimeInfo.hpp"

#include "InstrumentExtrae.hpp"
#include "InstrumentThreadLocalData.hpp"

#include <InstrumentThreadLocalDataSupport.hpp>
#include <InstrumentThreadLocalDataSupportImplementation.hpp>


// This is not defined in the extrae headers
extern "C" void Extrae_change_num_threads (unsigned n);


namespace Instrument {
	static unsigned int extrae_nanos_get_thread_id()
	{
		ThreadLocalData &threadLocal = getThreadLocalData();
		if (threadLocal._currentThreadId == thread_id_t()) {
			ExternalThreadLocalData &externalThreadLocalData = getExternalThreadLocalData();
			return externalThreadLocalData._currentThreadId;
		} else {
			return threadLocal._currentThreadId;
		}
	}
	
	static unsigned int extrae_nanos_get_virtual_cpu_or_external_thread_id()
	{
		ThreadLocalData &threadLocal = getThreadLocalData();
		if (threadLocal._currentThreadId == thread_id_t()) {
			ExternalThreadLocalData &externalThreadLocalData = getExternalThreadLocalData();
			return nanos_get_num_cpus() + externalThreadLocalData._currentThreadId;
		} else {
			return nanos_get_current_virtual_cpu();
		}
	}
	
	
	static unsigned int extrae_nanos_get_thread_id_for_initialization()
	{
		return 0;
	}
	
	static unsigned int extrae_nanos_get_virtual_cpu_or_external_thread_id_for_initialization()
	{
		return 0;
	}
	
	static unsigned int extrae_nanos_get_num_threads_for_initialization()
	{
		return 1;
	}
	
	static unsigned int extrae_nanos_get_num_cpus_and_external_threads_for_initialization()
	{
		return 1;
	}
	
	
	void initialize()
	{
		// This is a workaround to avoid an extrae segfault
		if ((getenv("EXTRAE_ON") == nullptr) && (getenv("EXTRAE_CONFIG_FILE") == nullptr)) {
			setenv("EXTRAE_ON", "1", 0);
		}
		
		RuntimeInfo::addEntry("instrumentation", "Instrumentation", "extrae");
		
		if (getenv("EXTRAE_CONFIG_FILE") != nullptr) {
			RuntimeInfo::addEntry("extrae_config_file", "Extrae Configuration File", getenv("EXTRAE_CONFIG_FILE"));
		}
		
		// Initial thread information callbacks
		// We set up a temporary thread_id function since the initialization calls
		// it (#@!?!) but the real one is not ready to be called yet
		if (_traceAsThreads) {
			Extrae_set_threadid_function ( extrae_nanos_get_thread_id_for_initialization );
			Extrae_set_numthreads_function ( extrae_nanos_get_num_threads_for_initialization );
			RuntimeInfo::addEntry("extrae_tracing_target", "Extrae Tracing Target", "thread");
		} else {
			Extrae_set_threadid_function ( extrae_nanos_get_virtual_cpu_or_external_thread_id_for_initialization );
			Extrae_set_numthreads_function ( extrae_nanos_get_num_cpus_and_external_threads_for_initialization );
			RuntimeInfo::addEntry("extrae_tracing_target", "Extrae Tracing Target", "cpu");
		}
		
		// Initialize extrae library
		Extrae_init();
		
		unsigned int zero = 0;
		
		Extrae_register_codelocation_type( _functionName, _codeLocation, (char *) "User Function Name", (char *) "User Function Location" );
		Extrae_define_event_type((extrae_type_t *) &_taskInstanceId, (char *) "Task instance", &zero, nullptr, nullptr);
		Extrae_define_event_type((extrae_type_t *) &_nestingLevel, (char *) "Task nesting level", &zero, nullptr, nullptr);
		
		// Register runtime states
		{
			unsigned int nval = NANOS_EVENT_STATE_TYPES;
			extrae_value_t values[nval];
			unsigned int i;
			
			for (i = 0; i < nval; i++) {
				values[i] = i;
			}
			Extrae_define_event_type(
				(extrae_type_t *) &_runtimeState, (char *) "Runtime state",
				&nval, values, (char **) _eventStateValueStr
			);
		}
		
		std::stringstream oss;
		unsigned extraeMajor, extraeMinor, extraeRevision;
		
		Extrae_get_version(&extraeMajor, &extraeMinor, &extraeRevision);
		oss << extraeMajor << "." << extraeMinor << "." << extraeRevision;
		RuntimeInfo::addEntry("extrae_version", "Extrae Version", oss.str());
		
		// Final thread information callbacks
		if (_traceAsThreads) {
			Extrae_set_threadid_function ( extrae_nanos_get_thread_id );
			Extrae_set_numthreads_function ( extrae_nanos_get_num_threads );
		} else {
			Extrae_set_threadid_function ( extrae_nanos_get_virtual_cpu_or_external_thread_id );
			Extrae_set_numthreads_function ( extrae_nanos_get_num_cpus_and_external_threads );
		}
	}
	
	
	void shutdown()
	{
		void *MPItrace_network_counters = dlsym(RTLD_DEFAULT, "MPItrace_network_counters");
		
		bool mustShutDownExtrae = true;
		if (MPItrace_network_counters != nullptr) {
			// Running under MPItrace
			
			typedef int MPI_Finalized_t(int *);
			MPI_Finalized_t *MPI_Finalized = (MPI_Finalized_t *) dlsym(RTLD_DEFAULT, "MPI_Finalized");
			if (MPI_Finalized != nullptr) {
				int finalized = 0;
				(*MPI_Finalized)(&finalized);
				
				mustShutDownExtrae = !finalized;
			} else {
				// Running under MPItrace but not an MPI program
			}
		}
		
		// Finalize extrae library
		if (mustShutDownExtrae) {
			Extrae_fini();
		}
	}
}


#endif // INSTRUMENT_EXTRAE_INIT_AND_SHUTDOWN_HPP
