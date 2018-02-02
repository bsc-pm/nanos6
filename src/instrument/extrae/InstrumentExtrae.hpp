/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_HPP
#define INSTRUMENT_EXTRAE_HPP

#include "PreloadedExtraeBouncer.hpp"

#include <nanos6.h>

#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/RWSpinLock.hpp"
#include "lowlevel/SpinLock.hpp"
#include "InstrumentThreadId.hpp"

#include <atomic>
#include <cstddef>
#include <list>
#include <map>
#include <set>
#include <string>
#include <vector>


namespace Instrument {
	struct ExtraeTaskInfoCompare {
		inline bool operator()(nanos_task_info *a, nanos_task_info *b) const;
	};
	
	typedef std::set<nanos_task_info *>              user_fct_map_t;
	
	extern const EnvironmentVariable<bool>           _traceAsThreads;
	extern const EnvironmentVariable<int>            _sampleBacktraceDepth;
	extern const EnvironmentVariable<long>           _sampleBacktracePeriod;
	extern const EnvironmentVariable<bool>           _sampleTaskCount;
	extern const EnvironmentVariable<bool>           _emitGraph;
	
	enum event_types {
		// OmpSs common
		_runtimeState = 9000000,     //!< Runtime state (extrae event type)
		
		_taskInstanceId = 9200002,
		_functionName = 9200011,     //!< Task function name
		_codeLocation = 9200021,     //!< Task code location
		_readyTasksEventType = 9200022,
		_liveTasksEventType = 9200023,
		_priorityEventType = 9200038,
		
		// Nanos6-specific (for now)
		_nestingLevel = 9500001,     //!< Nesting level
		
		// Common to extrae
		_samplingEventType = 30000000,
	};
	
	
	typedef enum { NANOS_NO_STATE, NANOS_NOT_RUNNING, NANOS_STARTUP, NANOS_SHUTDOWN, NANOS_ERROR, NANOS_IDLE,
						NANOS_RUNTIME, NANOS_RUNNING, NANOS_SYNCHRONIZATION, NANOS_SCHEDULING, NANOS_CREATION,
						NANOS_EVENT_STATE_TYPES
	} nanos_event_state_t;
	
	extern char const                               *_eventStateValueStr[];
	
	extern SpinLock _userFunctionMapLock;
	extern user_fct_map_t                            _userFunctionMap;
	
	extern SpinLock _backtraceAddressSetsLock;
	extern std::list<std::set<void *> *> _backtraceAddressSets;
	
	extern std::atomic<size_t> _nextTaskId;
	extern std::atomic<size_t> _readyTasks;
	extern std::atomic<size_t> _liveTasks;
	
	extern RWSpinLock _extraeThreadCountLock;
	
	extern int _externalThreadCount;
	
	enum dependency_tag_t {
		instanciation_dependency_tag = 0,
		strong_data_dependency_tag,
		weak_data_dependency_tag,
		control_dependency_tag
	};
	
	
	inline bool ExtraeTaskInfoCompare::operator()(nanos_task_info *a, nanos_task_info *b) const
	{
		std::string labelA(a->task_label != nullptr ? a->task_label : a->declaration_source);
		std::string labelB(b->task_label != nullptr ? b->task_label : b->declaration_source);
		
		if (labelA != labelB) {
			return (labelA < labelB);
		}
		std::string sourceA(a->declaration_source);
		std::string sourceB(b->declaration_source);
		
		if (sourceA != sourceB) {
			return (sourceA < sourceB);
		}
		
		return (a->run < b->run);
	}
	
	unsigned int extrae_nanos_get_num_threads();
	unsigned int extrae_nanos_get_num_cpus_and_external_threads();
	
	namespace Extrae {
		void lightweightDisableSamplingForCurrentThread();
		void lightweightEnableSamplingForCurrentThread();
	}
}

#endif
