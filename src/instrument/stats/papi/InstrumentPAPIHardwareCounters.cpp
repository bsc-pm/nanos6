/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

// Work around bug in PAPI header
#define ffsll papi_ffsll
#include <papi.h>
#undef ffsll

#include <algorithm>
#include <cassert>
#include <list>
#include <string>
#include <sstream>

#include "InstrumentPAPIHardwareCounters.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "system/RuntimeInfo.hpp"


namespace InstrumentHardwareCounters {
	namespace PAPI {
		int _initializationCount = 0;
		
		char const *cache_counting_strategy_descriptions[cache_counting_strategy_entries] =
		{
			"unavailable",
			"data hits and misses",
			"data accesses and hits",
			"data accesses and misses",
			"total hits and misses",
			"total accesses and hits",
			"total accesses and misses"
		};
		
		std::vector<int> _papiEventCodes;
		
		event_index_t _totalEvents = 0;
		event_index_t _l1CacheEventIndex = -1;
		event_index_t _l2CacheEventIndex = -1;
		event_index_t _l3CacheEventIndex = -1;
		event_index_t _FPInstructionsEventIndex = -1;
		event_index_t _referenceCyclesEventIndex = -1;
		
		cache_counting_strategy_t _l1CacheStrategy;
		cache_counting_strategy_t _l2CacheStrategy;
		cache_counting_strategy_t _l3CacheStrategy;
		
		
		static bool tryCombinedPAPIEvents(__attribute__((unused)) int eventSet)
		{
			return true;
		}
		
		template<typename... TS>
		static bool tryCombinedPAPIEvents(int eventSet, int event, TS... events)
		{
			// Check if the event is already in the list
			{
				auto it = std::find(_papiEventCodes.begin(), _papiEventCodes.end(), event);
				if (it != _papiEventCodes.end()) {
					return tryCombinedPAPIEvents(eventSet, events...);
				}
			}
			
			// Check the event
			int rc = PAPI_query_event(event);
			if (rc != PAPI_OK) {
				return false;
			}
			
			// Try to add the event
			rc = PAPI_add_event(eventSet, event);
			if (rc != PAPI_OK) {
				return false;
			}
			
			// Attempt to use the current event set
			rc = PAPI_start(eventSet);
			PAPI_stop(eventSet, nullptr);
			
			if (rc != PAPI_OK) {
				PAPI_remove_event(eventSet, event);
				return false;
			}
			
			// Recurse
			bool result = tryCombinedPAPIEvents(eventSet, events...);
			
			if (!result) {
				PAPI_remove_event(eventSet, event);
			}
			
			return result;
		}
		
		
		static void addEvents(__attribute__((unused)) int eventSet)
		{
		}
		
		template<typename... TS>
		static void addEvents(int eventSet, int event, TS... events)
		{
			// Check if the event is already in the list
			{
				auto it = std::find(_papiEventCodes.begin(), _papiEventCodes.end(), event);
				if (it == _papiEventCodes.end()) {
					_totalEvents++;
					_papiEventCodes.push_back(event);
				}
			}
			
			addEvents(eventSet, events...);
		}
		
		
		template<typename... TS>
		static bool tryToAddCombinedPAPIEvents(int eventSet, TS... events)
		{
			bool works = tryCombinedPAPIEvents(eventSet, events...);
			
			if (works) {
				addEvents(eventSet, events...);
			}
			
			return works;
		}
		
		
		template<typename... TS>
		static bool tryToAddCombinedPAPIEvents(int eventSet, /* OUT */ event_index_t &startEventIndex, TS... events)
		{
			bool works = tryCombinedPAPIEvents(eventSet, events...);
			
			if (works) {
				startEventIndex = _totalEvents;
				addEvents(eventSet, events...);
			}
			
			return works;
		}
		
		
		static void choosePAPIEventCodes()
		{
			assert(_totalEvents == 0);
			
			int eventSet = PAPI_NULL;
			int rc = PAPI_create_eventset(&eventSet);
			FatalErrorHandler::failIf(rc == PAPI_ENOMEM, "Not enough memory creating PAPI event set");
			FatalErrorHandler::failIf(rc == PAPI_EINVAL, "Invalid parameter creating PAPI event set");
			
			bool worked = tryToAddCombinedPAPIEvents(eventSet, PAPI_TOT_CYC, PAPI_TOT_INS);
			FatalErrorHandler::failIf(!worked, "Cannot count cycles or instruction with PAPI");
			
			// Find valid L2 cache events
			_l2CacheStrategy = hits_and_misses_cache_counting_strategy;
			worked = tryToAddCombinedPAPIEvents(eventSet, _l2CacheEventIndex, PAPI_L2_DCH, PAPI_L2_DCM);
			if (!worked) {
				_l2CacheStrategy = accesses_and_hits_cache_counting_strategy;
				worked = tryToAddCombinedPAPIEvents(eventSet, _l2CacheEventIndex, PAPI_L2_DCA, PAPI_L2_DCH);
			}
			if (!worked) {
				_l2CacheStrategy = accesses_and_misses_cache_counting_strategy;
				worked = tryToAddCombinedPAPIEvents(eventSet, _l2CacheEventIndex, PAPI_L2_DCA, PAPI_L2_DCM);
			}
			if (!worked) {
				_l2CacheStrategy = total_hits_and_misses_cache_counting_strategy;
				worked = tryToAddCombinedPAPIEvents(eventSet, _l2CacheEventIndex, PAPI_L2_TCH, PAPI_L2_TCM);
			}
			if (!worked) {
				_l2CacheStrategy = total_accesses_and_hits_cache_counting_strategy;
				worked = tryToAddCombinedPAPIEvents(eventSet, _l2CacheEventIndex, PAPI_L2_TCA, PAPI_L2_TCH);
			}
			if (!worked) {
				_l2CacheStrategy = total_accesses_and_misses_cache_counting_strategy;
				worked = tryToAddCombinedPAPIEvents(eventSet, _l2CacheEventIndex, PAPI_L2_TCA, PAPI_L2_TCM);
			}
			if (!worked) {
				_l2CacheStrategy = no_cache_counting_strategy;
			}
			
			// Find valid L3 cache events
			_l3CacheStrategy = hits_and_misses_cache_counting_strategy;
			worked = tryToAddCombinedPAPIEvents(eventSet, _l3CacheEventIndex, PAPI_L3_DCH, PAPI_L3_DCM);
			if (!worked) {
				_l3CacheStrategy = accesses_and_hits_cache_counting_strategy;
				worked = tryToAddCombinedPAPIEvents(eventSet, _l3CacheEventIndex, PAPI_L3_DCA, PAPI_L3_DCH);
			}
			if (!worked) {
				_l3CacheStrategy = accesses_and_misses_cache_counting_strategy;
				worked = tryToAddCombinedPAPIEvents(eventSet, _l3CacheEventIndex, PAPI_L3_DCA, PAPI_L3_DCM);
			}
			if (!worked) {
				_l3CacheStrategy = total_hits_and_misses_cache_counting_strategy;
				worked = tryToAddCombinedPAPIEvents(eventSet, _l3CacheEventIndex, PAPI_L3_TCH, PAPI_L3_TCM);
			}
			if (!worked) {
				_l3CacheStrategy = total_accesses_and_hits_cache_counting_strategy;
				worked = tryToAddCombinedPAPIEvents(eventSet, _l3CacheEventIndex, PAPI_L3_TCA, PAPI_L3_TCH);
			}
			if (!worked) {
				_l3CacheStrategy = total_accesses_and_misses_cache_counting_strategy;
				worked = tryToAddCombinedPAPIEvents(eventSet, _l3CacheEventIndex, PAPI_L3_TCA, PAPI_L3_TCM);
			}
			if (!worked) {
				_l3CacheStrategy = no_cache_counting_strategy;
			}
			
			// Find valid L1 cache vents
			_l1CacheStrategy = hits_and_misses_cache_counting_strategy;
			worked = tryToAddCombinedPAPIEvents(eventSet, _l1CacheEventIndex, PAPI_L1_DCH, PAPI_L1_DCM);
			if (!worked) {
				_l1CacheStrategy = accesses_and_hits_cache_counting_strategy;
				worked = tryToAddCombinedPAPIEvents(eventSet, _l1CacheEventIndex, PAPI_L1_DCA, PAPI_L1_DCH);
			}
			if (!worked) {
				_l1CacheStrategy = accesses_and_misses_cache_counting_strategy;
				worked = tryToAddCombinedPAPIEvents(eventSet, _l1CacheEventIndex, PAPI_L1_DCA, PAPI_L1_DCM);
			}
			if (!worked) {
				_l1CacheStrategy = total_hits_and_misses_cache_counting_strategy;
				worked = tryToAddCombinedPAPIEvents(eventSet, _l1CacheEventIndex, PAPI_L1_TCH, PAPI_L1_TCM);
			}
			if (!worked) {
				_l1CacheStrategy = total_accesses_and_hits_cache_counting_strategy;
				worked = tryToAddCombinedPAPIEvents(eventSet, _l1CacheEventIndex, PAPI_L1_TCA, PAPI_L1_TCH);
			}
			if (!worked) {
				_l1CacheStrategy = total_accesses_and_misses_cache_counting_strategy;
				worked = tryToAddCombinedPAPIEvents(eventSet, _l1CacheEventIndex, PAPI_L1_TCA, PAPI_L1_TCM);
			}
			if (!worked) {
				_l1CacheStrategy = no_cache_counting_strategy;
			}
			
			tryToAddCombinedPAPIEvents(eventSet, _FPInstructionsEventIndex, PAPI_FP_INS);
			tryToAddCombinedPAPIEvents(eventSet, _referenceCyclesEventIndex, PAPI_REF_CYC);
			
			rc = PAPI_cleanup_eventset(eventSet);
			FatalErrorHandler::failIf(rc != PAPI_OK, "cleaning PAPI training event set");
			
			rc = PAPI_destroy_eventset(&eventSet);
			FatalErrorHandler::failIf(rc != PAPI_OK, "destroying PAPI training event set");
		}
	} // InstrumentHardwareCounters::PAPI
	
	
	void initialize()
	{
		assert(PAPI::_initializationCount >= 0);
		PAPI::_initializationCount++;
		
		if (PAPI::_initializationCount > 1) {
			// Only really initialize once
			return;
		}
		
		int rc = PAPI_library_init(PAPI_VER_CURRENT);
		FatalErrorHandler::failIf(
			(rc > 0) && (rc != PAPI_VER_CURRENT),
			"Expected PAPI version ", PAPI_VER_CURRENT, " but got ", rc, " instead"
		);
		FatalErrorHandler::failIf(rc == PAPI_ENOMEM, "Not enough memory initializing PAPI");
		FatalErrorHandler::failIf(rc == PAPI_ECMP, "PAPI does not support this hardware");
		FatalErrorHandler::failIf(rc == PAPI_ESYS, "PAPI failed during initialization due to ", strerror(rc));
		
		rc = PAPI_thread_init((unsigned long (*)()) WorkerThread::getCurrentWorkerThread());
		FatalErrorHandler::failIf(rc != PAPI_OK, "PAPI failed during threading initialization");
		
		RuntimeInfo::addEntry("hardware_counters", "Hardware Counters", "PAPI");
		{
			std::ostringstream oss;
			
			int papiVersion = PAPI_get_opt(PAPI_LIB_VERSION, nullptr);
			oss << PAPI_VERSION_MAJOR(papiVersion) << "." << PAPI_VERSION_MINOR(papiVersion) << "." << PAPI_VERSION_REVISION(papiVersion);
			
			RuntimeInfo::addEntry("papi_version", "PAPI Version", oss.str());
		}
		
		PAPI::choosePAPIEventCodes();
		
		RuntimeInfo::addEntry(
			"l2_cache_hc_set", "L2 Cache Hardware Counter Set",
			PAPI::cache_counting_strategy_descriptions[PAPI::_l2CacheStrategy]
		);
		RuntimeInfo::addEntry(
			"l3_cache_hc_set", "L3 Cache Hardware Counter Set",
			PAPI::cache_counting_strategy_descriptions[PAPI::_l3CacheStrategy]
		);
		RuntimeInfo::addEntry(
			"l1_cache_hc_set", "L1 Cache Hardware Counter Set",
			PAPI::cache_counting_strategy_descriptions[PAPI::_l1CacheStrategy]
		);
		
		{
			std::vector<std::string> stringifiedCounterCodes;
			stringifiedCounterCodes.reserve(PAPI::_papiEventCodes.size());
			
			for (auto event : PAPI::_papiEventCodes) {
				char counterName[PAPI_MAX_STR_LEN];
				if (PAPI_event_code_to_name(event, counterName ) == PAPI_OK) {
					stringifiedCounterCodes.emplace_back(counterName);
				} else {
					std::ostringstream oss;
					oss << event;
					stringifiedCounterCodes.emplace_back(oss.str());
				}
			}
			
			RuntimeInfo::addListEntry("papi_events", "PAPI Event Names", stringifiedCounterCodes.begin(), stringifiedCounterCodes.end());
		}
	}
	
	
	void shutdown()
	{
		assert(PAPI::_initializationCount > 0);
		PAPI::_initializationCount--;
	}
	
} // InstrumentHardwareCounters

