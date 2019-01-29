/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "InstrumentReductions.hpp"
#include "../support/InstrumentThreadLocalDataSupport.hpp"
#include "InstrumentExtrae.hpp"


namespace Instrument {
    void receivedCompatibleReductionInfo(
            __attribute__((unused)) data_access_id_t dataAccessId,
            __attribute__((unused)) const ReductionInfo& reductionInfo,
            __attribute__((unused)) const InstrumentationContext &context) {
	}
	void deallocatedReductionInfo(
            __attribute__((unused)) data_access_id_t dataAccessId,
            __attribute__((unused)) const ReductionInfo *reductionInfo,
            __attribute__((unused)) const DataAccessRegion& originalRegion,
            __attribute__((unused)) const InstrumentationContext &context) {
	}
	
	void enterAllocateReductionInfo(
            __attribute__((unused)) data_access_id_t dataAccessId,
            __attribute__((unused)) const DataAccessRegion& accessRegion,
            __attribute__((unused)) const InstrumentationContext &context) {
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = (extrae_type_t) EventType::REDUCTION_STATE;
		ce.Values[0] = (extrae_value_t) NANOS_ALLOCATE_REDUCTION_INFO;
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	void exitAllocateReductionInfo(
            __attribute__((unused)) data_access_id_t dataAccessId,
            __attribute__((unused)) const ReductionInfo& reductionInfo,
            __attribute__((unused)) const InstrumentationContext &context) {
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = (extrae_type_t) EventType::REDUCTION_STATE;
		ce.Values[0] = (extrae_value_t) NANOS_OUTSIDE_REDUCTION;
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	void enterRetrievePrivateReductionStorage(
            __attribute__((unused)) const DataAccessRegion& originalRegion,
            __attribute__((unused)) const InstrumentationContext &context) {
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = (extrae_type_t) EventType::REDUCTION_STATE;
		ce.Values[0] = (extrae_value_t) NANOS_RETRIEVE_REDUCTION_STORAGE;
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	void exitRetrievePrivateReductionStorage(
            __attribute__((unused)) const ReductionInfo& reductionInfo,
            __attribute__((unused)) const DataAccessRegion& privateStorage,
            __attribute__((unused)) const DataAccessRegion& originalRegion,
            __attribute__((unused)) const InstrumentationContext &context) {
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = (extrae_type_t) EventType::REDUCTION_STATE;
		ce.Values[0] = (extrae_value_t) NANOS_OUTSIDE_REDUCTION;
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	void enterAllocatePrivateReductionStorage(
            __attribute__((unused)) const ReductionInfo& reductionInfo,
            __attribute__((unused)) const InstrumentationContext &context) {
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = (extrae_type_t) EventType::REDUCTION_STATE;
		ce.Values[0] = (extrae_value_t) NANOS_ALLOCATE_REDUCTION_STORAGE;
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	void exitAllocatePrivateReductionStorage(
            __attribute__((unused)) const ReductionInfo& reductionInfo,
            __attribute__((unused)) const DataAccessRegion& privateStorage,
            __attribute__((unused)) const InstrumentationContext &context) {
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = (extrae_type_t) EventType::REDUCTION_STATE;
		ce.Values[0] = (extrae_value_t) NANOS_OUTSIDE_REDUCTION;
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	void enterInitializePrivateReductionStorage(
            __attribute__((unused)) const ReductionInfo& reductionInfo,
            __attribute__((unused)) const DataAccessRegion& privateStorage,
            __attribute__((unused)) const InstrumentationContext &context) {
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = (extrae_type_t) EventType::REDUCTION_STATE;
		ce.Values[0] = (extrae_value_t) NANOS_INITIALIZE_REDUCTION_STORAGE;
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	void exitInitializePrivateReductionStorage(
            __attribute__((unused)) const ReductionInfo& reductionInfo,
            __attribute__((unused)) const DataAccessRegion& privateStorage,
            __attribute__((unused)) const InstrumentationContext &context) {
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = (extrae_type_t) EventType::REDUCTION_STATE;
		ce.Values[0] = (extrae_value_t) NANOS_OUTSIDE_REDUCTION;
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	void enterCombinePrivateReductionStorage(
            __attribute__((unused)) const ReductionInfo& reductionInfo,
            __attribute__((unused)) const DataAccessRegion& privateStorage,
            __attribute__((unused)) const DataAccessRegion& originalRegion,
            __attribute__((unused)) const InstrumentationContext &context) {
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = (extrae_type_t) EventType::REDUCTION_STATE;
		ce.Values[0] = (extrae_value_t) NANOS_COMBINE_REDUCTION_STORAGE;
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
	void exitCombinePrivateReductionStorage(
            __attribute__((unused)) const ReductionInfo& reductionInfo,
            __attribute__((unused)) const DataAccessRegion& privateStorage,
            __attribute__((unused)) const DataAccessRegion& originalRegion,
            __attribute__((unused)) const InstrumentationContext &context) {
		extrae_combined_events_t ce;
		
		ce.HardwareCounters = 1;
		ce.Callers = 0;
		ce.UserFunction = EXTRAE_USER_FUNCTION_NONE;
		ce.nEvents = 1;
		ce.nCommunications = 0;
		
		ce.Types  = (extrae_type_t *)  alloca (ce.nEvents * sizeof (extrae_type_t) );
		ce.Values = (extrae_value_t *) alloca (ce.nEvents * sizeof (extrae_value_t));
		
		ce.Types[0] = (extrae_type_t) EventType::REDUCTION_STATE;
		ce.Values[0] = (extrae_value_t) NANOS_OUTSIDE_REDUCTION;
		
		if (_traceAsThreads) {
			_extraeThreadCountLock.readLock();
		}
		ExtraeAPI::emit_CombinedEvents ( &ce );
		if (_traceAsThreads) {
			_extraeThreadCountLock.readUnlock();
		}
	}
}
