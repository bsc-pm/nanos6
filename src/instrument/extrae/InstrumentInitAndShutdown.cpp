/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <cstdlib>
#include <sstream>

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
		
		// Initialize extrae library
		Extrae_init();
		
		Extrae_register_codelocation_type( _functionName, _codeLocation, (char *) "User Function Name", (char *) "User Function Location" );
		Extrae_define_event_type(_taskInstanceId, "Task instance", 0, nullptr, nullptr);
		Extrae_define_event_type(_nestingLevel, "Task nesting level", 0, nullptr, nullptr);
		
		Extrae_define_event_type(_readyTasksEventType, "Number of ready tasks", 0, nullptr, nullptr);
		Extrae_define_event_type(_liveTasksEventType, "Number of live tasks", 0, nullptr, nullptr);
		
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
			extrae_value_t values[NANOS_EVENT_STATE_TYPES];
			unsigned int i;
			
			for (i = 0; i < NANOS_EVENT_STATE_TYPES; i++) {
				values[i] = i;
			}
			Extrae_define_event_type(_runtimeState, "Runtime state", NANOS_EVENT_STATE_TYPES, values, _eventStateValueStr);
		}
		
		// Thread information callbacks
		if (_traceAsThreads) {
			Extrae_set_threadid_function ( extrae_nanos_get_thread_id );
			Extrae_set_numthreads_function ( extrae_nanos_get_num_threads );
			Extrae_change_num_threads(extrae_nanos_get_num_threads());
			RuntimeInfo::addEntry("extrae_tracing_target", "Extrae Tracing Target", "thread");
		} else {
			Extrae_set_threadid_function ( extrae_nanos_get_virtual_cpu_or_external_thread_id );
			Extrae_set_numthreads_function ( extrae_nanos_get_num_cpus_and_external_threads );
			Extrae_change_num_threads(extrae_nanos_get_num_cpus_and_external_threads());
			RuntimeInfo::addEntry("extrae_tracing_target", "Extrae Tracing Target", "cpu");
		}
		
		// Force an event that allows to detect the trace as an OmpSs trace
		{
			extrae_combined_events_t ce;
			
			ce.HardwareCounters = 0;
			ce.Callers = 0;
			ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
			ce.nEvents = 1;
			ce.nCommunications = 0;
			
			ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
			ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
			
			ce.Types[0] = 9200001;
			ce.Values[0] = 0;
			
			Extrae_emit_CombinedEvents ( &ce );
		}
		
		_initialized = true;
		
		// Register any tracing point that arrived too early
		for (auto &p : _delayedNumericTracingPoints) {
			Extrae_define_event_type(_tracingPointBase + p.first._type, p.second.c_str(), 0, nullptr, nullptr);
		}
		for (auto &p : _delayedScopeTracingPoints) {
			extrae_value_t values[2] = {0, 1};
			char const *valueDescriptions[2] = {p.second._startDescription.c_str(), p.second._endDescription.c_str()};
			Extrae_define_event_type(_tracingPointBase + p.first._type, p.second._name.c_str(), 2, values, valueDescriptions);
		}
		for (auto &p : _delayedEnumeratedTracingPoints) {
			extrae_value_t values[p.second._valueDescriptions.size()];
			char const *extraeValueDescriptions[p.second._valueDescriptions.size()];
			
			for (size_t i = 0; i < p.second._valueDescriptions.size(); i++) {
				values[i] = i;
				extraeValueDescriptions[i] = p.second._valueDescriptions[i].c_str();
			}
			
			Extrae_define_event_type(
				_tracingPointBase + p.first._type, p.second._name.c_str(),
				p.second._valueDescriptions.size(), values, extraeValueDescriptions
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
		
		if (_sampleBacktraceDepth > 0) {
			Sampling::SigProf::setPeriod(_sampleBacktracePeriod);
			Sampling::SigProf::setHandler(&signalHandler);
			Sampling::SigProf::init();
			
			_profilingIsReady = true;
		}
	}
	
	
	void shutdown()
	{
		{
			std::lock_guard<SpinLock> guard(_userFunctionMapLock);
			for (nanos_task_info *taskInfo : _userFunctionMap) {
				std::string codeLocation = taskInfo->implementations[0].declaration_source;
				
				// Remove column
				codeLocation = codeLocation.substr(0, codeLocation.find_last_of(':'));
				
				std::string label;
				if (taskInfo->implementations[0].task_label != nullptr) {
					label = taskInfo->implementations[0].task_label;
				} else {
					label = codeLocation;
				}
				
				// Splice off the line number
				int lineNumber = 0;
				size_t linePosition = codeLocation.find_last_of(':');
				if (linePosition != std::string::npos) {
					std::istringstream iss(codeLocation.substr(linePosition+1));
					iss >> lineNumber;
					
					codeLocation.substr(0, linePosition);
				}
				
				Extrae_register_function_address (
					(void *) (taskInfo->implementations[0].run),
					label.c_str(),
					codeLocation.c_str(), lineNumber
				);
			}
		}
		
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

