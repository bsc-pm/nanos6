/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_THREAD_MANAGEMENT_HPP
#define INSTRUMENT_EXTRAE_THREAD_MANAGEMENT_HPP


#include "InstrumentExtrae.hpp"

#include "../api/InstrumentThreadManagement.hpp"
#include "../generic_ids/GenericIds.hpp"
#include "../support/InstrumentThreadLocalDataSupport.hpp"

#include <instrument/support/sampling/SigProf.hpp>


namespace Instrument {
	inline void enterThreadCreation(/* OUT */ thread_id_t &threadId, __attribute__((unused)) compute_place_id_t const &computePlaceId)
	{
		threadId = GenericIds::getNewThreadId();
		
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		if (_detailLevel > 0) {
			ce.nCommunications++;
		}
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		if (ce.nCommunications > 0) {
			ce.Communications = (extrae_user_communication_t *) alloca(sizeof(extrae_user_communication_t) * ce.nCommunications);
		}
		
		ce.Types[0] = (extrae_type_t) EventType::RUNTIME_STATE;
		ce.Values[0] = (extrae_value_t) NANOS_THREAD_CREATION;
		
		if (_detailLevel > 0) {
			ce.Communications[0].type = EXTRAE_USER_SEND;
			ce.Communications[0].tag = (extrae_comm_tag_t) thread_creation_tag;
			ce.Communications[0].size = 0;
			ce.Communications[0].partner = EXTRAE_COMM_PARTNER_MYSELF;
			ce.Communications[0].id = threadId;
		}
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	
	
	inline void exitThreadCreation(__attribute__((unused)) thread_id_t threadId)
	{
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = (extrae_type_t) EventType::RUNTIME_STATE;
		ce.Values[0] = (extrae_value_t) NANOS_IDLE;
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	
	inline void createdThread(thread_id_t threadId, __attribute__((unused)) compute_place_id_t const &computePlaceId)
	{
		ThreadLocalData &threadLocal = getThreadLocalData();
		threadLocal.init();
		
		threadLocal._nestingLevels.push_back(0);
		
		threadLocal._currentThreadId = threadId;
		
		if (_sampleBacktraceDepth > 0) {
			Sampling::SigProf::setUpThread(threadLocal);
			
			// We call the signal handler once since the first call to backtrace allocates memory.
			// If the signal is delivered within a memory allocation, the thread can deadlock.
			Sampling::SigProf::forceHandler();
		}
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.writeLock();
			ExtraeAPI::change_num_threads(extrae_nanos6_get_num_threads());
			_extraeThreadCountLock.writeUnlock();
		}
		
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		if (_detailLevel >= (int) THREADS_AND_CPUS_LEVEL) {
			if (_traceAsThreads) {
				ce.nEvents += 2; // CPU, THREAD_NUMA_NODE
			} else {
				ce.nEvents++; // THREAD
			}
		}
		
		if (_detailLevel > 0) {
			ce.nCommunications++;
		}
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		if (ce.nCommunications > 0) {
			ce.Communications = (extrae_user_communication_t *) alloca(sizeof(extrae_user_communication_t) * ce.nCommunications);
		}
		
		ce.Types[0] = (extrae_type_t) EventType::RUNTIME_STATE;
		ce.Values[0] = (extrae_value_t) NANOS_STARTUP;
		
		if (_detailLevel >= (int) THREADS_AND_CPUS_LEVEL) {
			if (_traceAsThreads) {
				ce.Types[1] = (extrae_type_t) EventType::CPU;
				ce.Values[1] = (extrae_value_t) (computePlaceId._id + 1);
				ce.Types[2] = (extrae_type_t) EventType::THREAD_NUMA_NODE;
				ce.Values[2] = (extrae_value_t) (computePlaceId._NUMANode + 1);
			} else {
				ce.Types[1] = (extrae_type_t) EventType::THREAD;
				ce.Values[1] = (extrae_value_t) (threadId + 1);
			}
		}
		
		if (_detailLevel > 0) {
			ce.Communications[0].type = EXTRAE_USER_RECV;
			ce.Communications[0].tag = (extrae_comm_tag_t) thread_creation_tag;
			ce.Communications[0].size = 0;
			ce.Communications[0].partner = EXTRAE_COMM_PARTNER_MYSELF;
			ce.Communications[0].id = threadId;
		}
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
		
		if (_sampleBacktraceDepth > 0) {
			_backtraceAddressSetsLock.lock();
			_backtraceAddressSets.push_back(&threadLocal._backtraceAddresses);
			_backtraceAddressSetsLock.unlock();
			
			Sampling::SigProf::enableThread(threadLocal);
		}
	}
	
	inline void precreatedExternalThread(/* OUT */ external_thread_id_t &threadId)
	{
		ExternalThreadLocalData &threadLocal = getExternalThreadLocalData();
		
		// Force the sentinel worker TLS to be initialized
		{
			ThreadLocalData &sentinelThreadLocal = getThreadLocalData();
			sentinelThreadLocal.init();
		}
		
		if (_traceAsThreads) {
			// Same thread counter as regular worker threads
			threadId = GenericIds::getCommonPoolNewExternalThreadId();
		} else {
			// Conter separated from worker threads
			threadId = GenericIds::getNewExternalThreadId();
		}
		
		threadLocal._currentThreadId = threadId;
	}
	
	template<typename... TS>
	void createdExternalThread(__attribute__((unused)) external_thread_id_t &threadId, __attribute__((unused)) TS... nameComponents)
	{
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = (extrae_type_t) EventType::RUNTIME_STATE;
		ce.Values[0] = (extrae_value_t) NANOS_IDLE;
		
		_extraeThreadCountLock.writeLock();
		if (_traceAsThreads) {
			ExtraeAPI::change_num_threads(extrae_nanos6_get_num_threads());
		} else {
			ExtraeAPI::change_num_threads(extrae_nanos6_get_num_cpus_and_external_threads());
		}
		_extraeThreadCountLock.writeUnlock();
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	
	inline void threadWillSuspend(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t cpu)
	{
		if (_traceAsThreads) {
			extrae_combined_events_t ce;
			
			ce.HardwareCounters = 0;
			ce.Callers = 0;
			ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
			ce.nEvents = 1;
			ce.nCommunications = 0;
			
			if (_detailLevel >= (int) THREADS_AND_CPUS_LEVEL) {
				if (_traceAsThreads) {
					ce.nEvents += 2; // CPU, THREAD_NUMA_NODE
				} else {
					ce.nEvents++; // THREAD
				}
			}
			
			ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
			ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
			
			ce.Types[0] = (extrae_type_t) EventType::RUNTIME_STATE;
			ce.Values[0] = (extrae_value_t) NANOS_NOT_RUNNING;
			
			if (_detailLevel >= (int) THREADS_AND_CPUS_LEVEL) {
				if (_traceAsThreads) {
					ce.Types[1] = (extrae_type_t) EventType::CPU;
					ce.Values[1] = (extrae_value_t) 0;
					ce.Types[2] = (extrae_type_t) EventType::THREAD_NUMA_NODE;
					ce.Values[2] = (extrae_value_t) 0;
				} else {
					ce.Types[1] = (extrae_type_t) EventType::THREAD;
					ce.Values[1] = (extrae_value_t) 0;
				}
			}
			
			ExtraeAPI::emit_CombinedEvents ( &ce );
		}
	}
	
	inline void threadHasResumed(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t cpu)
	{
		if (_traceAsThreads) {
			extrae_combined_events_t ce;
			
			ce.HardwareCounters = 0;
			ce.Callers = 0;
			ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
			ce.nEvents = 1;
			ce.nCommunications = 0;
			
			if (_detailLevel >= (int) THREADS_AND_CPUS_LEVEL) {
				if (_traceAsThreads) {
					ce.nEvents += 2; // CPU, THREAD_NUMA_NODE
				} else {
					ce.nEvents++; // THREAD
				}
			}
			
			ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
			ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
			
			ce.Types[0] = (extrae_type_t) EventType::RUNTIME_STATE;
			ce.Values[0] = (extrae_value_t) NANOS_IDLE;
			
			if (_detailLevel >= (int) THREADS_AND_CPUS_LEVEL) {
				if (_traceAsThreads) {
					ce.Types[1] = (extrae_type_t) EventType::CPU;
					ce.Values[1] = (extrae_value_t) (cpu._id + 1);
					ce.Types[2] = (extrae_type_t) EventType::THREAD_NUMA_NODE;
					ce.Values[2] = (extrae_value_t) (cpu._NUMANode + 1);
				} else {
					ce.Types[1] = (extrae_type_t) EventType::THREAD;
					ce.Values[1] = (extrae_value_t) (threadId + 1);
				}
			}
			
			ExtraeAPI::emit_CombinedEvents ( &ce );
		}
	}
	
	inline void threadWillSuspend(__attribute__((unused)) external_thread_id_t threadId)
	{
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 0;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = (extrae_type_t) EventType::RUNTIME_STATE;
		ce.Values[0] = (extrae_value_t) NANOS_NOT_RUNNING;
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	
	inline void threadHasResumed(__attribute__((unused)) external_thread_id_t threadId)
	{
			extrae_combined_events_t ce;
			
			ce.HardwareCounters = 0;
			ce.Callers = 0;
			ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
			ce.nEvents = 1;
			ce.nCommunications = 0;
			
			ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
			ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
			
			ce.Types[0] = (extrae_type_t) EventType::RUNTIME_STATE;
			ce.Values[0] = (extrae_value_t) NANOS_RUNTIME;
			
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	
	inline void threadWillShutdown()
	{
		if (_traceAsThreads) {
			extrae_combined_events_t ce;
			
			ce.HardwareCounters = 0;
			ce.Callers = 0;
			ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
			ce.nEvents = 1;
			ce.nCommunications = 0;
			
			if (_detailLevel >= (int) THREADS_AND_CPUS_LEVEL) {
				if (_traceAsThreads) {
					ce.nEvents += 2; // CPU, THREAD_NUMA_NODE
				} else {
					ce.nEvents++; // THREAD
				}
			}
			
			ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
			ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
			
			ce.Types[0] = (extrae_type_t) EventType::RUNTIME_STATE;
			ce.Values[0] = (extrae_value_t) NANOS_SHUTDOWN;
			
			if (_detailLevel >= (int) THREADS_AND_CPUS_LEVEL) {
				if (_traceAsThreads) {
					ce.Types[1] = (extrae_type_t) EventType::CPU;
					ce.Values[1] = (extrae_value_t) 0;
					ce.Types[2] = (extrae_type_t) EventType::THREAD_NUMA_NODE;
					ce.Values[2] = (extrae_value_t) 0;
				} else {
					ce.Types[1] = (extrae_type_t) EventType::THREAD;
					ce.Values[1] = (extrae_value_t) 0;
				}
			}
			
			ExtraeAPI::emit_CombinedEvents ( &ce );
		}
	}
	
	inline void threadEnterBusyWait(__attribute__((unused)) busy_wait_reason_t reason)
	{
	}
	
	inline void threadExitBusyWait()
	{
	}
}


#endif // INSTRUMENT_EXTRAE_THREAD_MANAGEMENT_HPP
