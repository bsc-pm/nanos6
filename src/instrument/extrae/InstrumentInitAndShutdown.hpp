/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_INIT_AND_SHUTDOWN_HPP
#define INSTRUMENT_EXTRAE_INIT_AND_SHUTDOWN_HPP

#include <cassert>
#include <cstdlib>

#include <nanos6/debug.h>

#include "../api/InstrumentInitAndShutdown.hpp"
#include "../generic_ids/GenericIds.hpp"
#include "system/RuntimeInfo.hpp"

#include "InstrumentExtrae.hpp"
#include "InstrumentThreadLocalData.hpp"

#include <InstrumentThreadLocalDataSupport.hpp>
#include <InstrumentThreadLocalDataSupportImplementation.hpp>


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
		return nanos_get_num_cpus();
	}
	
	static unsigned int extrae_nanos_get_num_threads_for_initialization()
	{
		return 1;
	}
	
	static unsigned int extrae_nanos_get_num_cpus_and_external_threads_for_initialization()
	{
		return nanos_get_num_cpus() + 1;
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
