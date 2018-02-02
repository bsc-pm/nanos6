/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <cstdlib>

#include <dlfcn.h>

#include <nanos6/debug.h>

#include "../api/InstrumentInitAndShutdown.hpp"
#include "../generic_ids/GenericIds.hpp"
#include "system/RuntimeInfo.hpp"

#include "InstrumentExtrae.hpp"
#include "InstrumentInitAndShutdown.hpp"
#include "InstrumentThreadId.hpp"
#include "InstrumentThreadLocalData.hpp"

#include <BacktraceWalker.hpp>
#include <CodeAddressInfo.hpp>
#include <InstrumentThreadLocalDataSupport.hpp>
#include <InstrumentThreadLocalDataSupportImplementation.hpp>
#include <instrument/support/sampling/SigProf.hpp>


// This is not defined in the extrae headers
extern "C" void Extrae_change_num_threads (unsigned n);


namespace Instrument {
	
	extern bool _profilingIsReady;
	
	
	namespace Extrae {
		void lightweightDisableSamplingForCurrentThread()
		{
			if (BacktraceWalker::involves_libc_malloc) {			
				Sampling::SigProf::lightweightDisableThread();
			} else {
				ThreadLocalData &threadLocal = getThreadLocalData();
				threadLocal._inMemoryAllocation++;
			}
		}
		
		void lightweightEnableSamplingForCurrentThread()
		{
			if (BacktraceWalker::involves_libc_malloc) {
				Sampling::SigProf::lightweightEnableThread();
			} else {
				ThreadLocalData &threadLocal = getThreadLocalData();
				threadLocal._inMemoryAllocation--;
				
				if ((threadLocal._disableCount > 0) || (threadLocal._lightweightDisableCount > 0)) {
					// Temporarily disabled
					return;
				}
				
				// Perform operations previously delayed because they involved memory allocations
				if (threadLocal._inMemoryAllocation == 0) {
					Sampling::SigProf::lightweightDisableThread();
					
					for (void *backlogAddress : threadLocal._backtraceAddressBacklog) {
						threadLocal._backtraceAddresses.insert(backlogAddress);
					}
					threadLocal._backtraceAddressBacklog.clear();
					
					Sampling::SigProf::lightweightEnableThread();
				}
			}
		}
	}
	
	
	static void signalHandler(Sampling::ThreadLocalData &samplingThreadLocal)
	{
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 0;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 0;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (_sampleBacktraceDepth * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (_sampleBacktraceDepth * sizeof (extrae_value_t));
		
		Instrument::ThreadLocalData &threadLocal = (Instrument::ThreadLocalData &) samplingThreadLocal;
		
		BacktraceWalker::walk(
			_sampleBacktraceDepth,
			/* Skip */ 3,
			[&](void *address, int currentFrame) {
				ce.Types[currentFrame] = _samplingEventType + currentFrame;
				ce.Values[currentFrame] = (extrae_value_t) address;
				
				if (threadLocal._inMemoryAllocation == 0) {
					threadLocal._backtraceAddresses.insert(address);
				} else if (threadLocal._backtraceAddressBacklog.size() + 1 < threadLocal._backtraceAddressBacklog.capacity()) {
					threadLocal._backtraceAddressBacklog.push_back(address);
				}
				
				ce.nEvents = currentFrame + 1;
			}
		);
		
		if (ce.nEvents == 0) {
			return;
		}
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		Extrae_emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	
	
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
		
		// Force the TLS to be initialized do avoid problems with the interception of malloc
		{
			ThreadLocalData &threadLocal = getThreadLocalData();
			threadLocal.init();
			__attribute__((unused)) ExternalThreadLocalData &externalThreadLocal = getExternalThreadLocalData();
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
		
		Extrae_define_event_type((extrae_type_t *) &_readyTasksEventType, (char *) "Number of ready tasks", &zero, nullptr, nullptr);
		Extrae_define_event_type((extrae_type_t *) &_liveTasksEventType, (char *) "Number of live tasks", &zero, nullptr, nullptr);
		
		// Register the events for the backtrace
		if (_sampleBacktraceDepth > 0) {
			for (int i = 0; i < _sampleBacktraceDepth; i++) {
				extrae_type_t functionEventType = _samplingEventType + i;
				extrae_type_t locationEventType = functionEventType + 100;
				
				std::ostringstream ossF, ossL;
				ossF << "Sampled functions (depth " << i << ")";
				ossL << "Sampled line functions (depth " << i << ")";
				Extrae_register_codelocation_type(
					functionEventType, locationEventType,
					(char *) ossF.str().c_str(), (char *) ossL.str().c_str()
				);
			}
		}
		
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
		
		
		{
			std::stringstream oss;
			unsigned extraeMajor, extraeMinor, extraeRevision;
			
			Extrae_get_version(&extraeMajor, &extraeMinor, &extraeRevision);
			oss << extraeMajor << "." << extraeMinor << "." << extraeRevision;
			RuntimeInfo::addEntry("extrae_version", "Extrae Version", oss.str());
			RuntimeInfo::addEntry("extrae_shared_object", "Extrae Shared Object", ExtraeSymbolResolverBase::getSharedObjectPath());
		}
		
		// Final thread information callbacks
		if (_traceAsThreads) {
			Extrae_set_threadid_function ( extrae_nanos_get_thread_id );
			Extrae_set_numthreads_function ( extrae_nanos_get_num_threads );
		} else {
			Extrae_set_threadid_function ( extrae_nanos_get_virtual_cpu_or_external_thread_id );
			Extrae_set_numthreads_function ( extrae_nanos_get_num_cpus_and_external_threads );
		}
		
		if (_sampleBacktraceDepth > 0) {
			Sampling::SigProf::setPeriod(_sampleBacktracePeriod);
			Sampling::SigProf::setHandler(&signalHandler);
			Sampling::SigProf::init();
			
			_profilingIsReady = true;
		}
	}
	
	
	void shutdown()
	{
		if (_sampleBacktraceDepth > 0) {
			// After this, on the next profiling signal, the corresponding timer gets disarmed
			Sampling::SigProf::disable();
			
			std::atomic_thread_fence(std::memory_order_seq_cst);
			
			std::set<void *> _alreadyProcessedAddresses;
			
			CodeAddressInfo::init();
			std::ostringstream functionList;
			std::ostringstream locationList;
			
			_backtraceAddressSetsLock.lock();
			for (auto addressSetPointer : _backtraceAddressSets) {
				for (void *address : *addressSetPointer) {
					if (_alreadyProcessedAddresses.find(address) != _alreadyProcessedAddresses.end()) {
						continue;
					}
					_alreadyProcessedAddresses.insert(address);
					
					CodeAddressInfo::Entry const &addressInfo = CodeAddressInfo::resolveAddress(address);
					functionList.str("");
					locationList.str("");
					
					bool first = true;
					for (auto const &frame : addressInfo._inlinedFrames) {
						CodeAddressInfo::FrameNames frameNames = CodeAddressInfo::getFrameNames(frame);
						
						if (frameNames._function.empty()) {
							// Could not retrieve the name of the function
							continue;
						}
						
						if (functionList.str().length() + 1 + frameNames._mangledFunction.length() > 2047) {
							break;
						}
						
						if (!first) {
							functionList << " ";
							locationList << " ";
						}
						
						functionList << frameNames._mangledFunction;
						locationList << frameNames._sourceLocation;
						first = false;
						
						// Extrae does not support attempting to push all the inlined functions into one string
						break;
					}
					
					if (functionList.str().empty()) {
						// Could not retrieve any function name
						continue;
					}
					
					Extrae_register_function_address (
						address,
						functionList.str().c_str(),
						locationList.str().c_str(), 0
					);
				}
				addressSetPointer->clear();
			}
			_backtraceAddressSetsLock.unlock();
			CodeAddressInfo::shutdown();
		}
		
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

