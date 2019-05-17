/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <cstdlib>
#include <mutex>
#include <sstream>

#include <dlfcn.h>

#include <nanos6/debug.h>

#include "../api/InstrumentInitAndShutdown.hpp"
#include "../generic_ids/GenericIds.hpp"
#include "system/RuntimeInfo.hpp"
#include "system/ompss/SpawnFunction.hpp"

#include "InstrumentExtrae.hpp"
#include "InstrumentInitAndShutdown.hpp"
#include "InstrumentThreadId.hpp"
#include "InstrumentThreadLocalData.hpp"

#include <BacktraceWalker.hpp>
#include <CodeAddressInfo.hpp>
#include <instrument/support/sampling/SigProf.hpp>


namespace Instrument {
	
	extern bool _profilingIsReady;
	
	
	namespace Extrae {
		bool lightweightDisableSamplingForCurrentThread()
		{
			if (BacktraceWalker::involves_libc_malloc) {			
				return Sampling::SigProf::lightweightDisableThread();
			} else {
				ThreadLocalData &threadLocal = getThreadLocalData();
				return (threadLocal._inMemoryAllocation++ == 0) && (threadLocal._disableCount == 0) && (threadLocal._lightweightDisableCount == 0);
			}
		}
		
		bool lightweightEnableSamplingForCurrentThread()
		{
			if (BacktraceWalker::involves_libc_malloc) {
				return Sampling::SigProf::lightweightEnableThread();
			} else {
				ThreadLocalData &threadLocal = getThreadLocalData();
				threadLocal._inMemoryAllocation--;
				
				if ((threadLocal._disableCount > 0) || (threadLocal._lightweightDisableCount > 0)) {
					// Temporarily disabled
					return false;
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
				
				return (threadLocal._inMemoryAllocation == 0) && (threadLocal._disableCount == 0) && (threadLocal._lightweightDisableCount == 0);
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
				ce.Types[currentFrame] = (int) EventType::SAMPLING + currentFrame;
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
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	
	
	static unsigned int extrae_nanos6_get_thread_id()
	{
		ThreadLocalData &threadLocal = getThreadLocalData();
		if (threadLocal._currentThreadId == thread_id_t()) {
			ExternalThreadLocalData &externalThreadLocalData = getExternalThreadLocalData();
			return externalThreadLocalData._currentThreadId;
		} else {
			return threadLocal._currentThreadId;
		}
	}
	
	static unsigned int extrae_nanos6_get_virtual_cpu_or_external_thread_id()
	{
		ThreadLocalData &threadLocal = getThreadLocalData();
		if (threadLocal._currentThreadId == thread_id_t()) {
			ExternalThreadLocalData &externalThreadLocalData = getExternalThreadLocalData();
			return nanos6_get_num_cpus() + externalThreadLocalData._currentThreadId;
		} else {
			return nanos6_get_current_virtual_cpu();
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
		ExtraeAPI::init();
		
		ExtraeAPI::register_codelocation_type((extrae_type_t) EventType::RUNNING_FUNCTION_NAME, (extrae_type_t) EventType::RUNNING_CODE_LOCATION, (char *) "Running Name", (char *) "Running Location");
		ExtraeAPI::register_codelocation_type((extrae_type_t) EventType::INSTANTIATING_FUNCTION_NAME, (extrae_type_t) EventType::INSTANTIATING_CODE_LOCATION, (char *) "Instantiating Name", (char *) "Instantiating Location");
		ExtraeAPI::define_event_type((extrae_type_t) EventType::TASK_INSTANCE_ID, "Task instance", 0, nullptr, nullptr);
		ExtraeAPI::define_event_type((extrae_type_t) EventType::NESTING_LEVEL, "Task nesting level", 0, nullptr, nullptr);
		
		ExtraeAPI::define_event_type((extrae_type_t) EventType::READY_TASKS, "Number of ready tasks", 0, nullptr, nullptr);
		ExtraeAPI::define_event_type((extrae_type_t) EventType::LIVE_TASKS, "Number of live tasks", 0, nullptr, nullptr);
		
		ExtraeAPI::define_event_type((extrae_type_t) EventType::PRIORITY, "Task priority", 0, nullptr, nullptr);
		
		if (_detailLevel >= (int) THREADS_AND_CPUS_LEVEL) {
			if (_traceAsThreads) {
				ExtraeAPI::define_event_type((extrae_type_t) EventType::CPU, "CPU", 0, nullptr, nullptr);
				ExtraeAPI::define_event_type((extrae_type_t) EventType::THREAD_NUMA_NODE, "NUMA Node", 0, nullptr, nullptr);
			} else {
				ExtraeAPI::define_event_type((extrae_type_t) EventType::THREAD, "Thread", 0, nullptr, nullptr);
			}
		}
		
		// Register the events for the backtrace
		if (_sampleBacktraceDepth > 0) {
			for (int i = 0; i < _sampleBacktraceDepth; i++) {
				extrae_type_t functionEventType = (extrae_type_t) EventType::SAMPLING + i;
				extrae_type_t locationEventType = functionEventType + 100;
				
				std::ostringstream ossF, ossL;
				ossF << "Sampled functions (depth " << i << ")";
				ossL << "Sampled line functions (depth " << i << ")";
				ExtraeAPI::register_codelocation_type(
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
			ExtraeAPI::define_event_type((extrae_type_t) EventType::RUNTIME_STATE, "Runtime state", NANOS_EVENT_STATE_TYPES, values, _eventStateValueStr);
		}
		
		// Register reduction states
		{
			extrae_value_t values[NANOS_REDUCTION_STATE_TYPES];
			unsigned int i;
			
			for (i = 0; i < NANOS_REDUCTION_STATE_TYPES; i++) {
				values[i] = i;
			}
			ExtraeAPI::define_event_type((extrae_type_t) EventType::REDUCTION_STATE, "Reduction state", NANOS_REDUCTION_STATE_TYPES, values, _reductionStateValueStr);
		}
		
		// Thread information callbacks
		if (_traceAsThreads) {
			ExtraeAPI::set_threadid_function ( extrae_nanos6_get_thread_id );
			ExtraeAPI::set_numthreads_function ( extrae_nanos6_get_num_threads );
			ExtraeAPI::change_num_threads(extrae_nanos6_get_num_threads());
			RuntimeInfo::addEntry("extrae_tracing_target", "Extrae Tracing Target", "thread");
		} else {
			ExtraeAPI::set_threadid_function ( extrae_nanos6_get_virtual_cpu_or_external_thread_id );
			ExtraeAPI::set_numthreads_function ( extrae_nanos6_get_num_cpus_and_external_threads );
			ExtraeAPI::change_num_threads(extrae_nanos6_get_num_cpus_and_external_threads());
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
			
			ExtraeAPI::emit_CombinedEvents ( &ce );
		}
		
		_initialized = true;
		
		// Register any tracing point that arrived too early
		for (auto &p : _delayedNumericTracingPoints) {
			ExtraeAPI::define_event_type((extrae_type_t) EventType::TRACING_POINT_BASE + p.first._type, p.second.c_str(), 0, nullptr, nullptr);
		}
		for (auto &p : _delayedScopeTracingPoints) {
			extrae_value_t values[2] = {0, 1};
			char const *valueDescriptions[2] = {p.second._startDescription.c_str(), p.second._endDescription.c_str()};
			ExtraeAPI::define_event_type((extrae_type_t) EventType::TRACING_POINT_BASE + p.first._type, p.second._name.c_str(), 2, values, valueDescriptions);
		}
		for (auto &p : _delayedEnumeratedTracingPoints) {
			extrae_value_t values[p.second._valueDescriptions.size()];
			char const *extraeValueDescriptions[p.second._valueDescriptions.size()];
			
			for (size_t i = 0; i < p.second._valueDescriptions.size(); i++) {
				values[i] = i;
				extraeValueDescriptions[i] = p.second._valueDescriptions[i].c_str();
			}
			
			ExtraeAPI::define_event_type(
				(extrae_type_t) EventType::TRACING_POINT_BASE + p.first._type, p.second._name.c_str(),
				p.second._valueDescriptions.size(), values, extraeValueDescriptions
			);
		}
		
		{
			std::stringstream oss;
			unsigned extraeMajor, extraeMinor, extraeRevision;
			
			ExtraeAPI::get_version(&extraeMajor, &extraeMinor, &extraeRevision);
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
			for (nanos6_task_info_t *taskInfo : _userFunctionMap) {
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
				
				void *runFunction = (void *) taskInfo->implementations[0].run;
				
				// Use the unique taskInfo address in case it is a spawned task
				if (SpawnedFunctions::isSpawned(taskInfo)) {
					runFunction = taskInfo;
				}
				
				ExtraeAPI::register_function_address (
					runFunction,
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
					
					ExtraeAPI::register_function_address (
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
		
		// Finalize extrae library
		ExtraeAPI::fini();
	}
}

