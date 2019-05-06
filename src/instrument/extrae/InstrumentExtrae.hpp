/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_HPP
#define INSTRUMENT_EXTRAE_HPP

#include "PreloadedExtraeBouncer.hpp"

#include <nanos6.h>

#include "InstrumentThreadId.hpp"
#include "InstrumentTracingPointTypes.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/RWSpinLock.hpp"
#include "lowlevel/SpinLock.hpp"
#include "system/ompss/SpawnFunction.hpp"

#include <atomic>
#include <cstddef>
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>


namespace Instrument {
	enum {
		THREADS_AND_CPUS_LEVEL = 2
	};
	
	struct ExtraeTaskInfoCompare {
		inline bool operator()(nanos6_task_info_t *a, nanos6_task_info_t *b) const;
	};
	
	typedef std::set<nanos6_task_info_t *>              user_fct_map_t;
	
	struct scope_tracing_point_info_t {
		std::string _name;
		std::string _startDescription;
		std::string _endDescription;
		
		scope_tracing_point_info_t(std::string const &name, std::string const &startDescription, std::string const &endDescription)
			: _name(name), _startDescription(startDescription), _endDescription(endDescription)
		{
		}
	};
	
	struct enumerated_tracing_point_info_t {
		std::string _name;
		std::vector<std::string> _valueDescriptions;
		
		enumerated_tracing_point_info_t(std::string const &name, std::vector<std::string> const &valueDescriptions)
			: _name(name), _valueDescriptions(valueDescriptions)
		{
		}
	};
	
	extern bool _initialized;
	extern std::map<tracing_point_type_t, std::string> _delayedNumericTracingPoints;
	extern std::map<tracing_point_type_t, scope_tracing_point_info_t> _delayedScopeTracingPoints;
	extern std::map<tracing_point_type_t, enumerated_tracing_point_info_t> _delayedEnumeratedTracingPoints;
	
	extern const EnvironmentVariable<bool> _traceAsThreads;
	extern const EnvironmentVariable<int> _sampleBacktraceDepth;
	extern const EnvironmentVariable<long> _sampleBacktracePeriod;
	extern const EnvironmentVariable<unsigned int> _detailLevel;
	
	enum struct EventType {
		// OmpSs common
			RUNTIME_STATE = 9000000,
			
			TASK_INSTANCE_ID = 9200002,
			RUNNING_FUNCTION_NAME = 9200011,
			RUNNING_CODE_LOCATION = 9200021,
			
			READY_TASKS = 9200022,
			LIVE_TASKS = 9200023,
			
			PRIORITY = 9200038,
			
			CPU = 9200042,
			THREAD_NUMA_NODE = 9200064,
			
		// [9500000:9699999] -- Nanos6-specific
			// 9500XXXX -- Nanos6 Tasking
				NESTING_LEVEL = 9500001,
				
				INSTANTIATING_FUNCTION_NAME = 9500002,
				INSTANTIATING_CODE_LOCATION = 9500003,
				
				THREAD = 9500004,
				
			// 9504XXX -- Reductions
				REDUCTION_STATE = 9504001,
				
			// 96XXXXX -- Tracing points
				TRACING_POINT_BASE = 9600000,
		
		// Common to extrae
			SAMPLING = 30000000,
	};
	
	
	typedef enum { NANOS_NO_STATE, NANOS_NOT_RUNNING, NANOS_STARTUP, NANOS_SHUTDOWN, NANOS_ERROR, NANOS_IDLE,
						NANOS_RUNTIME, NANOS_RUNNING, NANOS_SYNCHRONIZATION, NANOS_SCHEDULING, NANOS_CREATION,
						NANOS_THREAD_CREATION,
						NANOS_EVENT_STATE_TYPES
	} nanos6_event_state_t;
	
	extern char const                               *_eventStateValueStr[];
	
	typedef enum { NANOS_OUTSIDE_REDUCTION, NANOS_ALLOCATE_REDUCTION_INFO,
						NANOS_RETRIEVE_REDUCTION_STORAGE, NANOS_ALLOCATE_REDUCTION_STORAGE,
						NANOS_INITIALIZE_REDUCTION_STORAGE, NANOS_COMBINE_REDUCTION_STORAGE,
						NANOS_REDUCTION_STATE_TYPES
	} nanos6_reduction_state_t;
	
	extern char const                               *_reductionStateValueStr[];
	
	extern SpinLock _userFunctionMapLock;
	extern user_fct_map_t                            _userFunctionMap;
	
	extern SpinLock _backtraceAddressSetsLock;
	extern std::list<std::set<void *> *> _backtraceAddressSets;
	
	extern std::atomic<size_t> _nextTaskId;
	extern std::atomic<size_t> _readyTasks;
	extern std::atomic<size_t> _liveTasks;
	extern std::atomic<size_t> _nextTracingPointKey;
	
	extern RWSpinLock _extraeThreadCountLock;
	
	extern int _externalThreadCount;
	
	enum dependency_tag_t {
		instantiation_dependency_tag = 0xffffff00,
		strong_data_dependency_tag,
		weak_data_dependency_tag,
		control_dependency_tag,
		thread_creation_tag
	};
	
	
	inline bool ExtraeTaskInfoCompare::operator()(nanos6_task_info_t *a, nanos6_task_info_t *b) const
	{
		std::string labelA(a->implementations[0].task_label != nullptr ? a->implementations[0].task_label : a->implementations[0].declaration_source);
		std::string labelB(b->implementations[0].task_label != nullptr ? b->implementations[0].task_label : b->implementations[0].declaration_source);
		
		if (labelA != labelB) {
			return (labelA < labelB);
		}
		std::string sourceA(a->implementations[0].declaration_source);
		std::string sourceB(b->implementations[0].declaration_source);
		
		if (sourceA != sourceB) {
			return (sourceA < sourceB);
		}
		
		void *runA = SpawnedFunctions::isSpawned(a) ? (void *) a : (void *) a->implementations[0].run;
		void *runB = SpawnedFunctions::isSpawned(b) ? (void *) b : (void *) b->implementations[0].run;
		
		return (runA < runB);
	}
	
	unsigned int extrae_nanos6_get_num_threads();
	unsigned int extrae_nanos6_get_num_cpus_and_external_threads();
	
	namespace Extrae {
		// Returns true if this is the call that actually disables it
		bool lightweightDisableSamplingForCurrentThread();
		
		// Returns true is this is the call that actually reenables it
		bool lightweightEnableSamplingForCurrentThread();
	}
}

#endif
